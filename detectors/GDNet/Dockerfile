FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

ARG DETECTOR_DIR=GDNet

WORKDIR /detector

COPY common common

COPY $DETECTOR_DIR .

RUN pip install -U pip \
    && pip install --no-cache-dir gdown -r requirements.txt \
    && pip install git+https://github.com/true-real-michael/dss_crf \
    && cd ..

RUN gdown "1xOwuH9lWizGVnPmEH77_81Sp9EAN054O"

ENTRYPOINT ["python", "run.py"]
