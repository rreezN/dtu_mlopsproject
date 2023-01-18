from fastapi import FastAPI, UploadFile, File, Form
from google.cloud import storage
import timm
import torch
from PIL import Image
from io import BytesIO
import numpy as np
from torchvision import transforms
import pickle
from model import MyAwesomeConvNext

app = FastAPI()

BUCKET_NAME = "dtumlops2023"
MODEL_FILE = "local_model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
encoded_state_dict = blob.download_as_string()

# because model downloaded into string, need to convert it back
buff = BytesIO(encoded_state_dict)

# state_dict = torch.load(buff, map_location=torch.device('cpu'))
# model = timm.create_model(
#     "convnext_atto",
#     pretrained=True,
#     in_chans=3,
#     num_classes=10,
# )
my_model = MyAwesomeConvNext.load_from_checkpoint(buff)
my_model.eval()

index2animal = {
    0: 'a cat',
    1: 'a cow',
    2: 'a dog',
    3: 'an elephant',
    4: 'a gorilla',
    5: 'a hippo',
    6: 'a monkey',
    7: 'a panda',
    8: 'a tiger',
    9: 'a zebra'
}

@app.post("/")
async def read_root(file: UploadFile = File(...)):
    data = BytesIO(await file.read())
    image = Image.open(data)
    if image.mode != "RGB":
        image = image.convert("RGB")
    imgshape = np.array(image).shape
    image = image.resize((224, 224))

    T = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            )
        ]
    )

    norm_image = T(image)
    prediction = my_model(norm_image.unsqueeze(0))[0]
    print(prediction)
    most_likely = prediction.argmax().item()
    animal = index2animal[most_likely]
    certainties = prediction.exp()/prediction.exp().sum()
    certainty = certainties[most_likely].item()*100
    return f"I am {certainty:.3f}% certain that the image is {animal}"
