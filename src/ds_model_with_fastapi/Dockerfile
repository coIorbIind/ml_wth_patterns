FROM python:3.10

COPY src/ /code/src
WORKDIR /code/src

RUN pip install -r ds_model_with_fastapi/requirements.txt

ENV PYTHONPATH "/code/src"

RUN echo "$PYTHONPATH"

CMD ["uvicorn", "ds_model_with_fastapi.main:get_app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--factory"]