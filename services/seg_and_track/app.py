from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.segmentor import Segmentor

app = FastAPI()

segmentor = Segmentor()

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    result = segmentor.segment_image(file.file)
    return JSONResponse(content=result.dict())
