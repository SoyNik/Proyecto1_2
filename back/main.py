import json
import pickle
import fastapi
import numpy as np
import pandas as pd
from typing import Optional
from keras.models import load_model
from dataModel import *
from joblib import load
from fastapi.middleware.cors import CORSMiddleware


app = fastapi.FastAPI(title= "Proyecto 1-2", description="Desarrollado por: Nicolás Segura Castro", version="1.0.2")

origins = ["http://localhost:4200/consulta"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
 return {}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
 return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = DataModel.columns('self')
    model = load("./notebooks/assets/modelo.pkl")
    model.named_steps['model'].model = load_model('./notebooks/assets/keras_model.h5')
    result = model.predict_proba(df)
    lista = (np.argmax(result)+1).tolist()
    json_prediccion = json.dumps(lista)
    lista_2 = result.tolist()[0]
    json_probabilidades = json.dumps(lista_2)
    return {"predict": json_prediccion, "probabilities": json_probabilidades}
