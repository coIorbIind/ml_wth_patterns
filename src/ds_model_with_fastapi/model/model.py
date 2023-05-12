import cv2
import torch
import numpy as np
from pickle import UnpicklingError
from torchvision.models import resnet50, ResNet50_Weights

from ds_model_with_fastapi.config.config import settings
from .train import train


class MetaSingleton(type):
    """Метакласс, определяющий поведение singleton'a"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Model(metaclass=MetaSingleton):
    """Класс DS модели"""
    def __init__(self):
        self.data_prep_pipeline = settings.PREP_PIPELINE
        # Load model from disk or train new
        self.cnn_model = Model.load_model(settings.MODEL_PATH)

        if self.cnn_model is None:
            self.cnn_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            for item in self.cnn_model.parameters():
                item.requires_grad = False

            self.cnn_model.fc = torch.nn.Linear(self.cnn_model.fc.in_features, 2)
            self.cnn_model.fc.requires_grad = True

            self.train()

    def train(self):
        """Обучение модели"""
        print('Started training')
        
        self.cnn_model = train(self.cnn_model, self.data_prep_pipeline)
        torch.save(self.cnn_model, settings.MODEL_PATH)

        print('Finished training')

    def predict(self, img: np.ndarray) -> int:
        """Предсказание результата"""
        prep_img = self.data_prep_pipeline.preprocess(img)

        prep_img = torch.unsqueeze(prep_img, 0)

        cnn_output = self.cnn_model(prep_img)
        pred_class = torch.argmax(cnn_output, dim=1).item()

        return pred_class
        
    @classmethod
    def load_model(cls, classifier_path: str) -> torch.nn.Module | None:
        """Считывание модели с диска"""
        try:
            result = torch.load(classifier_path, map_location=torch.device('cpu'))
        except (FileNotFoundError, IsADirectoryError, UnpicklingError, EOFError):
            result = None

        return result
    
    def dump_model(self, target_path: str) -> None:
        torch.save(self.cnn_model, target_path)


class FileModelAdapter:
    def __init__(self, model_class: Model):
        self.model = model_class
        self.num_class_dict = {
            0: 'CAT',
            1: 'DOG',
        }

    def predict(self, file_obj) -> str:
        vect_img = cv2.imdecode(file_obj)

        return self.num_class_dict[self.model.predict(vect_img)]
    
    def train(self) -> None:
        self.model.train()


def get_model():
    return Model()
