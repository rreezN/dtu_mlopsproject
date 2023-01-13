FROM gcr.io/deeplearning-platform-release/pytorch-gpu

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/

WORKDIR /
RUN pip install pip --upgrade
RUN pip install -r requirements.txt --no-cache-dir