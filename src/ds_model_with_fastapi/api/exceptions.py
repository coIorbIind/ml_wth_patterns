from fastapi import Request
from fastapi.responses import JSONResponse


class BaseAPIException(Exception):
    """ Базовое исключение """
    status_code = 500
    code = 'exception'
    message = 'Ошибка!'

    def __init__(self, loc=None, **kwargs):
        self.context = kwargs
        self.loc = [['body'] + location for location in loc] if loc else [['body']]

    def to_json(self):
        return {
            'code': self.code,
            'context': self.context,
            'detail': [
                {
                    'location': loc,
                    'message': self.context.get('message', self.message)
                } for loc in self.loc
            ]
        }


def exception_handler(request: Request, exception: BaseAPIException) -> JSONResponse:
    """ Перевод исключения в json """
    return JSONResponse(
        status_code=exception.status_code,
        content=exception.to_json()
    )
