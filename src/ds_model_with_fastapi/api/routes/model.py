from fastapi import Depends
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

from ds_model_with_fastapi.model.model import get_model, Model
from ds_model_with_fastapi.api.schemas.model import RequestModelSchema, ResponseModelSchema


router = InferringRouter()


@cbv(router)
class ModelRoute:
    model: Model = Depends(get_model)

    @router.post('/predict')
    def predict(self, data: RequestModelSchema) -> ResponseModelSchema:
        """ Метод предсказания результата по пользовательским данным """
        result = self.model.predict()
        return ResponseModelSchema(prediction=result)
