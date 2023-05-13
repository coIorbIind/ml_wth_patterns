from fastapi import Depends, UploadFile
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from checksumdir import dirhash

from ds_model_with_fastapi.model.model import get_model, FileModelAdapter
from ds_model_with_fastapi.api.schemas.model import ResponseModelSchema, TrainModelSchema
from ds_model_with_fastapi.config.config import settings


router = InferringRouter()


@cbv(router)
class ModelRoute:
    model: FileModelAdapter = Depends(get_model)

    @router.post('/predict')
    def predict(self, file: UploadFile) -> ResponseModelSchema:
        """ Метод предсказания результата по пользовательским данным """
        result = self.model.predict(file.file)
        print(result)
        return ResponseModelSchema(prediction=result)

    @router.post('/train')
    def train(self) -> TrainModelSchema:
        """Метод запуска обучения модели"""
        temp_hash = dirhash(settings.TRAIN_SETTINGS.TRAIN_DATA_DIR, 'md5')
        if temp_hash != self.model.model.directory_hash:
            self.model.train()
            return TrainModelSchema(trained=True)
        return TrainModelSchema(trained=False)
