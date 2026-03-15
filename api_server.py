from fastapi import FastAPI, UploadFile, File
import shutil

from predict_fault import predict_from_audio
from llm.explain_fault import explain_fault

app = FastAPI()

@app.post("/detect")

async def detect_engine_fault(file: UploadFile = File(...)):

    file_location = "temp.wav"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    fault = predict_from_audio(file_location)

    explanation = explain_fault(fault)

    return {
        "fault": fault,
        "explanation": explanation
    }
