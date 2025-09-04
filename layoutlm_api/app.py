# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import os

from inference import setup_cfg, run_batch_prediction
from detectron2.engine import DefaultPredictor

class ImagePathList(BaseModel):
    image_paths: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🚀 启动时加载模型
    config_file = "cascade_layoutlmv3.yaml"
    # change the path to your pretrained weight
    weights = "/path/to/pretrian_layoutlmv3_weight.pth"
    cfg = setup_cfg(config_file, weights)
    app.state.predictor = DefaultPredictor(cfg)
    print("✅ 模型已加载到 GPU 并准备推理。")

    yield 
    print("🛑 FastAPI 正在关闭")

# 使用 lifespan 初始化模型
app = FastAPI(lifespan=lifespan)

@app.post("/predict_batch/")
def predict_batch(data: ImagePathList):
    predictor = app.state.predictor
    image_paths = data.image_paths
    missing = [path for path in image_paths if not os.path.exists(path)]
    if missing:
        return {"error": f"The following image paths do not exist: {missing}"}

    results = run_batch_prediction(predictor, image_paths)
    return results