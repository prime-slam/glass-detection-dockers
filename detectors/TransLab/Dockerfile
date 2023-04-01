FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /detector

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -U pip && pip install --no-cache-dir gdown -r requirements.txt

RUN gdown "1shxxcNHmpu8TmLosT1qmFnojqwQzM22e"

RUN python setup.py develop

ENTRYPOINT ["python", "run.py"]