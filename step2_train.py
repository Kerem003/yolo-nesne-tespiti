"""
ADIM 2 - MODEL EĞİTİMİ (YOLOv8s, transfer learning)

Model  : YOLOv8s (CSPDarknet backbone + PAN-FPN neck + anchor-free decoupled head)
         COCO üzerinde ön-eğitimli ağırlıklarla başlatılır, Pascal VOC'un 20 sınıfı için
         ince ayar (fine-tuning) yapılır. Çıkış katmanı 80 -> 20 sınıfa yeniden kurulur.

Hiperparametreler (seçim gerekçeleri raporda):
  imgsz=640, epochs=30, batch=16, optimizer=AdamW, lr0=0.001, cosine LR,
  weight_decay=0.0005, warmup_epochs=3, AMP (mixed precision), erken durdurma patience=10
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

HPARAMS = dict(
    data=str(ROOT / "voc.yaml"),
    imgsz=640,
    epochs=30,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,            # son LR = lr0 * lrf
    cos_lr=True,
    momentum=0.937,      # AdamW için beta1
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=10,
    amp=True,
    workers=4,
    seed=42,
    deterministic=False,
    project=str(ROOT / "runs"),
    name="yolov8s_voc",
    exist_ok=True,
    plots=True,
)

if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "YOK (CPU)")

    model = YOLO("yolov8s.pt")  # COCO ön-eğitimli ağırlıklar
    t0 = time.time()
    model.train(**HPARAMS)
    train_sec = time.time() - t0

    info = {
        "model": "YOLOv8s",
        "pretrained": "COCO (yolov8s.pt)",
        "train_time_sec": round(train_sec, 1),
        "train_time_min": round(train_sec / 60, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch": torch.__version__,
        "hyperparameters": {k: v for k, v in HPARAMS.items() if k not in ("project", "name", "exist_ok", "plots", "data")},
    }
    (RESULTS / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"Eğitim tamamlandı: {train_sec/60:.1f} dk")
