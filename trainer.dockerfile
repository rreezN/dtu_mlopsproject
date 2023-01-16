FROM gcr.io/deeplearning-platform-release/pytorch-cpu

COPY requirements_train.txt requirements_train.txt
COPY setup.py setup.py
COPY .dvc/ .dvc/
COPY src/ src/

WORKDIR /
RUN pip install pip --upgrade
RUN pip install -r requirements_train.txt --no-cache-dir
