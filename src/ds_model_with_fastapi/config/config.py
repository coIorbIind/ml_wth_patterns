from ds_model_with_fastapi.model.utils import PreprocessingPipeline


class TrainSettings:
    EPOCH_COUNT = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_WORKERS = 2
    RANDOM_STATE = 1234
    DEVICE = 'cpu'
    TRAIN_DATA_DIR = 'ds_model_with_fastapi/model/data/train'
    TEST_DATA_DIR = 'ds_model_with_fastapi/model/data/test'
    RUNS_LOG_DIR = 'ds_model_with_fastapi/model/runs'


class Settings:
    MODEL_PATH = 'ds_model_with_fastapi/model/model.bin'
    PREP_PIPELINE = PreprocessingPipeline()
    TRAIN_SETTINGS = TrainSettings()


settings = Settings()
