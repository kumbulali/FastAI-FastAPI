from fastai.vision.all import *
from PIL import Image
import requests
from fastapi import FastAPI

path = untar_data(URLs.PETS)/'images'


app = FastAPI()


pil_img = Image.open(requests.get('https://i.pinimg.com/280x280_RS/9e/34/d5/9e34d57b5824d1e23f439b89d8a10242.jpg', stream=True).raw)
print(type(pil_img))
img = PILImage.create(np.array(pil_img.convert('RGB')))
print(type(img))

