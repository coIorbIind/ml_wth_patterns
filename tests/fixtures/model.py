

class ModelMock:

    def predict(self):
        return 'fake prediction'


def override_get_model():
    return ModelMock()
