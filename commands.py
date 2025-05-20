import os
import joblib
from multiprocessing import Process
from threading import Lock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel
from typing import List, Optional, Literal, Any
from multiprocessing import Value


class request_fit(BaseModel):
    name: str
    X: List[List[float]]
    y: List[Any]
    model_type: Literal['logreg', 'randf', 'lr']
    params: Optional[dict] = {}

class request_pred(BaseModel):
    name: str
    X: List[List[float]]

class model_inf(BaseModel):
    name: str


classes = {
    'logreg': LogisticRegression,
    'randf': RandomForestClassifier,
    'lr' : LinearRegression
}

class ModeCommands:
    def __init__(self, model_dir, max_processes, max_loaded):
        self.model_dir = model_dir
        self.max_processes = max_processes - 1  
        self.max_loaded = max_loaded
        self.loaded_models = {}
        self.lock = Lock()
        self.active_processes = Value('i', 0)
        
        os.makedirs(model_dir, exist_ok=True)
        


        # print("Текущая рабочая директория:", os.getcwd())
        # print("Сохраняем модели в:", self.model_dir)

    def _save(self, model, name):
        os.makedirs(self.model_dir, exist_ok=True)
        file = os.path.join(self.model_dir, f"{name}.joblib")
        joblib.dump(model, file)
        

    def _load(self, name):
        path = os.path.join(self.model_dir, f"{name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {name} not found.")
        return joblib.load(path)

    def _fit(self, X, y, model_type, params, name):
        model = classes[model_type](**params)
        model.fit(X, y)
        self._save(model, name)
        with self.lock:
            self.active_processes.value -= 1

    def fit(self, name, X, y, model_type, params):
        with self.lock:
            if self.active_processes.value >= self.max_processes:
                raise RuntimeError("Wait, all the cores are busy.")
            if os.path.exists(os.path.join(self.model_dir, f"{name}.joblib")):
                raise ValueError("Model already exists.")
            self.active_processes.value += 1
        p = Process(target=self._fit, args=(X, y, model_type, params, name))
        p.start()
        return {"status": "started"}

    def load(self, name):
        if name in self.loaded_models:
            return {"status": "already_loaded"}
        if len(self.loaded_models) >= self.max_loaded:
            raise RuntimeError(f"You can't load more than {self.max_loaded} models.")
        model = self._load(name)
        self.loaded_models[name] = model
        return {"status": "loaded"}

    def unload(self, name):
        if name in self.loaded_models:
            del self.loaded_models[name]
            return {"status": "unloaded"}
        raise ValueError("Model not found.")

    def predict(self, name, X):
        if name not in self.loaded_models:
            raise ValueError("Model not loaded.")
        return self.loaded_models[name].predict(X).tolist()

    def remove(self, name):
        path = os.path.join(self.model_dir, f"{name}.joblib")
        if os.path.exists(path):
            os.remove(path)
            return {"status": "deleted"}
        raise FileNotFoundError("Model not found.")

    def remove_all(self):
        for file in os.listdir(self.model_dir):
            if file.endswith(".joblib"):
                os.remove(os.path.join(self.model_dir, file))
        return {"status": "all_deleted"}
