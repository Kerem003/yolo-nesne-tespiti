"""
ADIM 3 - PERFORMANS DEĞERLENDİRMESİ (TEST kümesi: VOC2007 test, eğitimde görülmedi)

Metrikler:
  - mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1   (ultralytics val, split='test')
  - Sınıf bazında AP@0.5
  - Ortalama IoU: her gerçek kutu için aynı sınıftaki en iyi tahminin IoU'su (conf>=0.25)
  - Tahmin süresi: ön-işleme / çıkarım / son-işleme (ms/görüntü) + FPS
  - Eğitim eğrileri (loss, mAP) grafiği
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RUN = ROOT / "runs" / "yolov8s_voc"
WEIGHTS = RUN / "weights" / "best.pt"
TEST_IMG = ROOT / "datasets" / "VOC" / "images" / "test"
TEST_LBL = ROOT / "datasets" / "VOC" / "labels" / "test"


def box_iou(a, b):
    """a: (N,4), b: (M,4) xyxy -> (N,M)"""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def mean_iou(model, n_images=1000):
    """Gerçek kutu başına, aynı sınıftan en iyi eşleşen tahminin IoU ortalaması."""
    imgs = sorted(TEST_IMG.glob("*.jpg"))[:n_images]
    ious, matched, total = [], 0, 0
    for p in imgs:
        # Görüntüler tek tek verilir: liste olarak verilince ultralytics hepsini tek batch yapıp GPU belleğini aşıyor.
        r = model.predict(str(p), conf=0.25, imgsz=640, verbose=False)[0]
        h, w = r.orig_shape
        lines = (TEST_LBL / f"{p.stem}.txt").read_text().split("\n")
        gt = np.array([list(map(float, l.split())) for l in lines if l.strip()])
        if len(gt) == 0:
            continue
        gcls = gt[:, 0].astype(int)
        gxyxy = np.stack([(gt[:, 1] - gt[:, 3] / 2) * w, (gt[:, 2] - gt[:, 4] / 2) * h,
                          (gt[:, 1] + gt[:, 3] / 2) * w, (gt[:, 2] + gt[:, 4] / 2) * h], 1)
        pxyxy = r.boxes.xyxy.cpu().numpy()
        pcls = r.boxes.cls.cpu().numpy().astype(int)
        for c in np.unique(gcls):
            g = gxyxy[gcls == c]
            pr = pxyxy[pcls == c]
            total += len(g)
            if len(pr) == 0:
                continue
            best = box_iou(g, pr).max(1)
            ious.extend(best[best > 0.5].tolist())   # eşleşme kabul eşiği IoU>0.5
            matched += int((best > 0.5).sum())
    return float(np.mean(ious)), matched / max(total, 1), len(imgs)


def plot_training_curves():
    df = pd.read_csv(RUN / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, lab in [("train/box_loss", "train box"), ("val/box_loss", "val box"),
                   ("train/cls_loss", "train cls"), ("val/cls_loss", "val cls"),
                   ("train/dfl_loss", "train dfl"), ("val/dfl_loss", "val dfl")]:
        ax[0].plot(df["epoch"], df[k], label=lab, ls="-" if "train" in k else "--")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(fontsize=8)
    ax[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
    ax[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    ax[1].set_title("Doğrulama mAP"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    ax[2].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    ax[2].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    ax[2].set_title("Doğrulama Precision / Recall"); ax[2].set_xlabel("Epoch"); ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "training_curves.png", dpi=150)
    plt.close()


def plot_per_class(names, ap50):
    order = np.argsort(ap50)
    plt.figure(figsize=(8, 7))
    plt.barh([names[i] for i in order], ap50[order], color="#4C72B0")
    for y, v in enumerate(ap50[order]):
        plt.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
    plt.xlim(0, 1.05); plt.xlabel("AP@0.5"); plt.title("Sınıf bazında AP@0.5 (VOC2007 test)")
    plt.tight_layout()
    plt.savefig(RESULTS / "per_class_ap50.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS))

    # 1) Test kümesinde standart dedektör metrikleri
    m = model.val(data=str(ROOT / "voc.yaml"), split="test", imgsz=640, batch=16,
                  project=str(ROOT / "runs"), name="test_eval", exist_ok=True, plots=True)
    p, r = float(m.box.mp), float(m.box.mr)
    names = model.names
    ap50 = np.zeros(len(names))
    for i, c in enumerate(m.box.ap_class_index):
        ap50[c] = m.box.ap50[i]

    # 2) Ortalama IoU
    miou, match_rate, n_iou = mean_iou(model)

    # 3) Tahmin süresi (tek görüntü, batch=1, GPU ısındıktan sonra)
    imgs = [str(p_) for p_ in sorted(TEST_IMG.glob("*.jpg"))[:220]]
    for x in imgs[:20]:
        model.predict(x, imgsz=640, verbose=False)
    pre = inf = post = 0.0
    t0 = time.time()
    for x in imgs[20:]:
        s = model.predict(x, imgsz=640, verbose=False)[0].speed
        pre += s["preprocess"]; inf += s["inference"]; post += s["postprocess"]
    wall = (time.time() - t0) / 200 * 1000
    n = 200

    out = {
        "split": "VOC2007 test",
        "n_images": int(m.seen) if hasattr(m, "seen") else 4952,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r + 1e-9), 4),
        "mean_iou_matched": round(miou, 4),
        "gt_match_rate_iou50": round(match_rate, 4),
        "iou_eval_images": n_iou,
        "per_class_ap50": {names[i]: round(float(ap50[i]), 4) for i in range(len(names))},
        "speed_ms": {"preprocess": round(pre / n, 2), "inference": round(inf / n, 2),
                     "postprocess": round(post / n, 2), "end_to_end_wall": round(wall, 2)},
        "fps_end_to_end": round(1000 / wall, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    (RESULTS / "test_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    plot_training_curves()
    plot_per_class(names, ap50)
    print(json.dumps(out, indent=2, ensure_ascii=False))
