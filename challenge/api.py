import fastapi
from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import logging
from typing import List, Dict
from challenge.model import DelayModel
from challenge.data_validation import validate_column_names, validate_categorical_values
import uvicorn

app = fastapi.FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

delay_model = DelayModel()

class PredictionRequest(BaseModel):
    data: List[Dict]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> dict:
    """
    Endpoint for generating predictions based on input data.
    """
    try:
        data = request.data
        print(data)
        input_data = pd.DataFrame(request.data)
        delay_model.run_data_validations(input_data)
        features = delay_model.preprocess(input_data)
        predictions = delay_model.predict(features)
        return {"predict": predictions}
    except ValueError as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

@app.post("/retrain", status_code=200)
async def retrain_model() -> dict:
    """
    Endpoint for retraining the model with new data.
    """
    try:
        delay_model.run_training_pipeline()
        return {"status": "Model retrained and stored in memory successfully."}
    except Exception as e:
        logging.error(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during model retraining.")


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)