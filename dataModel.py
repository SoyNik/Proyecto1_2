from pydantic import BaseModel
from typing import List

class DataModel(BaseModel):

 medical_abstracts: str

 def columns(self):
    return ["medical_abstracts"] 

# Clase DataModelList
class DataModelList(BaseModel):

    data: List[DataModel]

class DataEsperada(BaseModel):

    problems_described:float

    def columns(self):
        return ["problems_described"]

class DataEsperadaLista(BaseModel):
    dataEsperada : List[DataEsperada]