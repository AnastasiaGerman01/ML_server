from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
from commands import ModeCommands, request_fit, request_pred, model_inf

load_dotenv()

app = FastAPI()

server = ModeCommands(
    model_dir=os.getenv("MODEL_DIR"),
    max_processes=int(os.getenv("MAX_PROCESSES_ALLOWED")),
    max_loaded=int(os.getenv("MAX_LOADED_MODELS"))
)


@app.post("/fit")
def fit(req: request_fit):
    try:
        return server.fit(req.name, req.X, req.y, req.model_type, req.params or {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(req: request_pred):
    try:
        
        return {"predictions": server.predict(req.name, req.X)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/load")
def load(cfg: model_inf):
    try:
        return server.load(cfg.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/unload")
def unload(cfg: model_inf):
    try:
        return server.unload(cfg.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/remove")
def remove(cfg: model_inf):
    try:
        return server.remove(cfg.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/remove_all")
def remove_all():
    return server.remove_all()
