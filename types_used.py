from pydantic import BaseModel
from typing import List, Optional, Literal, Any

class request_fit(BaseModel):
    name: str
    X: List[List[float]]
    y: List[Any]
    model_type: Literal['logreg', 'rf']
    params: Optional[dict] = {}

class request_pred(BaseModel):
    name: str
    X: List[List[float]]

class model_inf(BaseModel):
    name: str
