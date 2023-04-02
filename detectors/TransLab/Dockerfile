FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ARG DETECTOR_DIR=TransLab

WORKDIR /detector

COPY ../common common

COPY $DETECTOR_DIR .

RUN apt-get update && apt-get install g++ ffmpeg libsm6 libxext6 -y

RUN pip install -U pip && pip install --no-cache-dir gdown -r requirements.txt

RUN gdown "1shxxcNHmpu8TmLosT1qmFnojqwQzM22e"

RUN python setup.py develop

ENTRYPOINT ["python", "run.py"]