from apis.routers.models import Config,Dataset
from fastapi import UploadFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import tensorflow as tf
from PIL import Image
import io
from fastapi.responses import FileResponse,StreamingResponse
import cv2

config=Config()

async def segment_image_return_result(file: UploadFile, model):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")  

    target_size = (256, 256)
    image = image.resize(target_size)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_mask = model.predict(img_array)[0]
    pred_mask = np.squeeze(pred_mask)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    mask_img = Image.fromarray(pred_mask * 255).convert("RGB")

    combined_width = image.width + mask_img.width
    combined_img = Image.new("RGB", (combined_width, image.height))
    combined_img.paste(image, (0, 0))
    combined_img.paste(mask_img, (image.width, 0))

    img_io = io.BytesIO()
    combined_img.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(
        img_io,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=segmentation_result.png"}
    )