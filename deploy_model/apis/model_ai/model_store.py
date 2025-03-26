from bm25s.hf import BM25HF
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model = None 

async def load_model_local():
    path_model="../models/best_dc-unet_model.h5"
    model=load_model(path_model)
    
    return model


def set_model(new_model):
    global model
    model = new_model

def get_model():
    global model
    if model is None:
        raise ValueError("Model has not been loaded yet!")
    return model