from pydantic import BaseModel


class RequestModelSchema(BaseModel):
    """Схема данных, которые прислыет пользователь"""
    pass


class ResponseModelSchema(BaseModel):
    """Схема данных, которые получает пользователь"""
    prediction: str | None


class TrainModelSchema(BaseModel):
    """Схема данных, которые возвращаются при запросе на обучение"""
    trained: bool
