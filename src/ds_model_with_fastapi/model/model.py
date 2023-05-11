from ds_model_with_fastapi.config.config import settings


class MetaSingleton(type):
    """Метакласс, определяющий поведение singleton'a"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Model(metaclass=MetaSingleton):
    """Класс DS модели"""

    def train(self):
        """Обучение модели"""
        pass

    def predict(self):
        """Предсказание результата"""
        pass

    @classmethod
    def load_model(cls):
        """Считывание модели с диска"""
        pass


def get_model():
    return Model()
