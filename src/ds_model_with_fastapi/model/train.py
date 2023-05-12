import os
import cv2
import numpy as np
import random
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

from ds_model_with_fastapi.config.config import settings
from .utils import PreprocessingPipeline


class DogsCatsDataset(Dataset):
    def __init__(self, data_folder: str, prep_pipeline: PreprocessingPipeline):
        self.prep_pipeline = prep_pipeline

        cats_path = os.path.join(data_folder, 'Cats')
        dogs_path = os.path.join(data_folder, 'Dogs')

        images_classes_list = [(os.path.join(cats_path, cat_img), 0) for cat_img in os.listdir(cats_path)] + [(os.path.join(dogs_path, dog_img), 1) for dog_img in os.listdir(dogs_path)]
        
        random.shuffle(images_classes_list)

        self.shuffled_images_classes_list = images_classes_list

    def __getitem__(self, index):
        raw_img = cv2.imread(self.shuffled_images_classes_list[index][0])

        return {
            'image': self.prep_pipeline.preprocess(raw_img),
            'label': self.shuffled_images_classes_list[index][1],
        }

    def __len__(self):
        return len(self.shuffled_images_classes_list)

        
def seed_all():   
    RANDOM_SEED = settings.TRAIN_SETTINGS.RANDOM_STATE 
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    torch.cuda.manual_seed_all(RANDOM_SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model: nn.Module, prep_pipeline: PreprocessingPipeline) -> nn.Module:
    seed_all()

    train_settings = settings.TRAIN_SETTINGS
    tb_writer = SummaryWriter(os.path.join(train_settings.RUNS_LOG_DIR, datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
    
    device = torch.device(train_settings.DEVICE)

    model.to(device)

    train_dataset = DogsCatsDataset(train_settings.TRAIN_DATA_DIR, prep_pipeline)
    test_dataset = DogsCatsDataset(train_settings.TEST_DATA_DIR, prep_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=train_settings.BATCH_SIZE, num_workers=train_settings.NUM_WORKERS, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=train_settings.BATCH_SIZE, num_workers=train_settings.NUM_WORKERS, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=train_settings.LEARNING_RATE)
    loss = nn.CrossEntropyLoss()

    for temp_epoch in range(train_settings.EPOCH_COUNT):
        model.train()

        train_acc_loss = 0
        # Train
        for temp_batch in train_loader:
            img_batch = temp_batch['image'].to(device)
            target_batch = temp_batch['label'].to(device)
            
            output = model(img_batch)

            temp_loss = loss(output, target_batch)

            optimizer.zero_grad()
            temp_loss.backward()
            optimizer.step()

            train_acc_loss += temp_loss.item()
            break

        train_acc_loss /= len(train_loader)
        tb_writer.add_scalar("Loss/train", train_acc_loss, temp_epoch + 1)

        test_acc_loss = 0
        test_answers_list = list()
        test_prediction_list = list()
        
        # Test
        with torch.no_grad():
            for temp_batch in test_loader:
                img_batch = temp_batch['image'].to(device)
                target_batch = temp_batch['label'].to(device)
                
                output = model(img_batch)

                temp_loss = loss(output, target_batch)
                test_acc_loss += temp_loss.item()

                temp_pred = torch.argmax(output, dim=1)
                test_prediction_list += temp_pred.tolist()
                test_answers_list += target_batch.tolist()
                break

            test_acc_loss /= len(test_loader)

            tb_writer.add_scalar("Loss/test", test_acc_loss, temp_epoch + 1)

            # Log metrics
            class_report_dict = classification_report(test_answers_list, test_prediction_list, output_dict=True, target_names=['Cat', 'Dog'], zero_division=0)

            for class_name in ['Dog', 'Cat']:
                temp_metrics_dict = class_report_dict[class_name]

                tb_writer.add_scalar(f'{class_name}/Precision', temp_metrics_dict['precision'], temp_epoch + 1)
                tb_writer.add_scalar(f'{class_name}/Recall', temp_metrics_dict['recall'], temp_epoch + 1)
                tb_writer.add_scalar(f'{class_name}/F1', temp_metrics_dict['f1-score'], temp_epoch + 1)

    return model
