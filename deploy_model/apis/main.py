from fastapi import FastAPI
from apis.routers import views
from apis.model_ai import model_store
import asyncio
import os
import time


app=FastAPI()

async def initialize_resources():
    try:
        start=time.time()
        print("Loading model ...")
        model=await model_store.load_model_local()
        model_store.set_model(model)
        print(f"Model loaded in {time.time() - start:.2f} seconds.")
        
    except Exception as e:
        print("Error while initialize resouce: ",e)
        

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(initialize_resources()) # running but not block main flow
    
app.include_router(views.router)