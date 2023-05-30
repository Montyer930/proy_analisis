from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Define la estructura de los datos de entrada
class PredictionInput(BaseModel):
    Platform: int
    Year_of_Release: int
    Genre: int
    Publisher: int
    Critic_Score: float
    Developer: int

app = FastAPI()

# Carga el modelo entrenado
model = joblib.load("modelo_red.joblib")

# Define la ruta POST para hacer la predicción
@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Convierte los datos de entrada a un array numpy
    input_array = np.array([[
        input_data.Platform,
        input_data.Year_of_Release,
        input_data.Genre,
        input_data.Publisher,
        input_data.Critic_Score,
        input_data.Developer
    ]])
    
    # Realiza la predicción utilizando el modelo cargado
    prediction = model.predict(input_array)
    
    # Retorna el resultado de la predicción
    return {"prediction": prediction[0]}

# Punto de entrada del servidor de desarrollo de FastAPI
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
