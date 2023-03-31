FROM python:3.7-bullseye
WORKDIR /detector

COPY . .

RUN pip install -U pip \
    && pip install --no-cache-dir gdown -r requirements.txt \
    && gdown "1shxxcNHmpu8TmLosT1qmFnojqwQzM22e" \
    && python setup.py develop

ENTRYPOINT ["python", "run.py"]