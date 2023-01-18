FROM gcr.io/deeplearning-platform-release/pytorch-gpu

# # COPY requirements_train.txt requirements_train.txt
# # COPY setup.py setup.py
# # COPY src/ src/

# # WORKDIR /
# # RUN pip install pip --upgrade
# # RUN pip install -r requirements_train.txt --no-cache-dir
# ---------------------------------------------------------------
# FROM python:3.8

# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_train.txt requirements_train.txt
COPY setup.py setup.py
COPY src/ src/


WORKDIR /
RUN pip install pip --upgrade
RUN pip install --ignore-installed -r requirements_train.txt --no-cache-dir

ENTRYPOINT ["python3", "-u", "src/models/train_model.py"]

