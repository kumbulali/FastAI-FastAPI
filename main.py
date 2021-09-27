from fastai.vision.all import *
from PIL import Image
import requests
from fastapi import FastAPI

path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224), num_workers= 0)

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

app = FastAPI()


@app.get("/kedi/")
async def read_user_me(url: str):
    pil_img = Image.open(requests.get(url, stream=True).raw)
    img = PILImage.create(np.array(pil_img.convert('RGB')))
    is_cat, _, probs = learn.predict(img)
    return {"image_url": url,
            "Is this a cat? ": is_cat,
            "Probability it's a cat: ": "{:.6f}".format(probs[1].item())}
