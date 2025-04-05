from tensorflow.keras.models import load_model
from huggingface_hub import login, hf_hub_download
from dotenv import load_dotenv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

load_dotenv()
MODEL_REPO=os.getenv("path_model_dcunet")
ACCESS_TOKEN_HF=os.getenv("access_token_read")

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

async def load_or_download_model():
    global model
    try:
        if model is None:
            print("This is REPO MODEL: ", MODEL_REPO)
            print("Loading model from Hugging Face...")
            model_path=hf_hub_download(MODEL_REPO,"best_dc-unet_model.h5")
            model=load_model(model_path)
            print("Model loaded successfully.")
    except Exception as e:
        print("Error in load or download model: ", e)
    return model