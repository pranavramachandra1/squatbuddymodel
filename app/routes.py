from fastapi import FastAPI, Query, HTTPException, APIRouter, UploadFile, File
from typing import List
import numpy as np
import cv2
import io
from PIL import Image
import base64
from pydantic import BaseModel
from typing import List

class Frames(BaseModel):
    frames: List[str]

# local packages:
from .services.squat_buddy import SquatBuddy

sb = SquatBuddy()

router = APIRouter()
# router.mount("/static", StaticFiles(directory="static"), name="static")

# Parges:
@router.get("/")
def root():
    return {"message": "Hello World"}

@router.post("/predict_batch")
async def predict_batch(data: Frames):
    try:
        results = []
        for base64_str in data.frames:
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image)

            # Predict using your SquatBuddy instance
            results.append(list(sb.predict(image)))

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))