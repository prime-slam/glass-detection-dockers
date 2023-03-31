FROM python:3.9-bullseye
WORKDIR /detector

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y \
    && pip install -U pip \
    && pip install --no-cache-dir gdown -r requirements.txt \
    && gdown "1shxxcNHmpu8TmLosT1qmFnojqwQzM22e" \
    && python setup.py develop

ENTRYPOINT ["python", "run.py"]