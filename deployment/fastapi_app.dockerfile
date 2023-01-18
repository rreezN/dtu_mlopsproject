FROM python:3.9-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install google-cloud-storage
RUN pip install torchvision
RUN pip install timm
RUN pip install fastapi
RUN pip install Pillow
RUN pip install numpy
RUN pip install python-multipart
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install pytorch-lightning

COPY fastapi_app.py fastapi_app.py
COPY model.py model.py

CMD exec uvicorn fastapi_app:app --port $PORT --host 0.0.0.0 --workers 1
