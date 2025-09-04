import torch
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from typing import List
from ditod import add_vit_config  # from your project
import os

def setup_cfg(config_file: str, weights: str) -> "CfgNode":
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()
    return cfg

def predict_image(image_path: str,
                  config_file: str = "cascade_layoutlmv3.yaml",
                  weights: str = "model_final.pth") -> dict:
    """
    对单张图像进行预测并返回结果
    """
    cfg = setup_cfg(config_file, weights)
    predictor = DefaultPredictor(cfg)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image from {image_path}")

    outputs = predictor(image)

    # 可视化结果（可选）
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 保存可视化图像（可选）
    output_path = os.path.splitext(image_path)[0] + "_pred.jpg"
    cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])

    # 结构化输出结果
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "visualized_path": output_path,
    }


def run_batch_prediction(predictor: DefaultPredictor, image_paths: list) -> list:
    metadata = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0]) if len(predictor.cfg.DATASETS.TRAIN) else None
    results = []

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            results.append({"image_path": image_path, "error": "Failed to load"})
            continue

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        vis = Visualizer(image[:, :, ::-1], metadata, scale=1.0).draw_instance_predictions(instances)
        vis_path = os.path.splitext(image_path)[0] + "_pred.jpg"
        cv2.imwrite(vis_path, vis.get_image()[:, :, ::-1])

        results.append({
            "image_path": image_path,
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "visualized_path": vis_path,
        })

    return results


if __name__ == "__main__":
    pass
    # change the params below, and test
    # result = predict_image("path/to/image",
    #                        config_file="cascade_layoutlmv3.yaml",
    #                        weights="/path/to/pretrian_layoutlmv3_weight.pth")
    # print(result)