import numpy as np
import cv2
from torchvision.transforms import ToTensor


class PreprocessingPipeline:
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        convertion_steps = [
            ConvertSizeStep(640),
            ConvertColorStep(),
            ConvertToTensor(),
        ]

        result = data

        for temp_step in convertion_steps:
            result = temp_step.preprocess(result)

        return result


class ConvertSizeStep(PreprocessingPipeline):
    def __init__(self, img_size: int):
        self.img_size = img_size

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return cv2.resize(data, (self.img_size, self.img_size))


class ConvertColorStep(PreprocessingPipeline):
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)


class ConvertToTensor(PreprocessingPipeline):
    def __init__(self):
        self.to_tensor = ToTensor()

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return self.to_tensor(data)
