from fastapi import APIRouter, File, UploadFile,status
from fastapi.responses import JSONResponse, FileResponse 
from apis.model_ai.model_store import get_model
from apis.routers.utils import segment_image_return_result


router=APIRouter()

@router.get("/ping_to_server")
def ping_to_server():
    return {"message":"This is server segmentation medical"}


@router.post("/predict_each_image")
async def segmentation_each_image(file: UploadFile = File(...)):
    try:
        dc_unet_model=get_model()
        return await segment_image_return_result(file, dc_unet_model) 
    except Exception as e:
        print(f"Error in predict sentence API: {e}")