FROM gcr.io/deeplearning-platform-release/pytorch-gpu

COPY requirements_train.txt requirements_train.txt
COPY setup.py setup.py
COPY src/ src/

WORKDIR /
RUN pip install pip --upgrade
RUN pip install -r requirements_train.txt --no-cache-dir