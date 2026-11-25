"""
ADIM 5 - KENDİ TELEFON FOTOĞRAFLARIMIZLA TEST

1) Telefonla çekilen 5 fotoğrafı  my_photos/  klasörüne koyun.
2) my_photos/ground_truth.json dosyasına her fotoğrafta GERÇEKTE bulunan nesneleri yazın, ör:
       {"foto1.jpg": {"person": 2, "chair": 1}, "foto2.jpg": {"cat": 1}}
   (Sınıf adları: aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
    diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor)
3) Çalıştırın. Çıktılar:
       results/real_photos/<foto>_pred.jpg     -> kutulu tahmin görselleri
       results/real_photos_grid.png            -> 5 sonucun birleşik görseli
       results/real_photos.json                -> tahmin vs gerçek karşılaştırması
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from ultralytics import YOLO

ROOT = Path(__file__).parent
PHOTOS = ROOT / "my_photos"
OUT = ROOT / "results" / "real_photos"
CONF = 0.25

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "runs" / "yolov8s_voc" / "weights" / "best.pt"))
    gt_file = PHOTOS / "ground_truth.json"
    gt_all = json.loads(gt_file.read_text(encoding="utf-8")) if gt_file.exists() else {}

    photos = sorted(p for p in PHOTOS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    report, panels = [], []
    for p in photos:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # telefon EXIF yönünü düzelt
        r = model.predict(img, conf=CONF, imgsz=640, verbose=False)[0]
        plotted = r.plot()[:, :, ::-1]
        Image.fromarray(plotted).save(OUT / f"{p.stem}_pred.jpg", quality=92)
        panels.append((p.name, plotted))

        pred = Counter(model.names[int(c)] for c in r.boxes.cls)
        gt = Counter(gt_all.get(p.name, {}))
        classes = sorted(set(pred) | set(gt))
        tp = sum(min(pred[c], gt[c]) for c in classes)
        fp = sum(max(pred[c] - gt[c], 0) for c in classes)
        fn = sum(max(gt[c] - pred[c], 0) for c in classes)
        report.append({
            "photo": p.name,
            "predicted": dict(pred),
            "ground_truth": dict(gt),
            "TP": tp, "FP": fp, "FN": fn,
            "detections": [{"class": model.names[int(b.cls)], "conf": round(float(b.conf), 3)} for b in r.boxes],
            "inference_ms": round(r.speed["inference"], 1),
        })
        print(f"{p.name}: tahmin={dict(pred)}  gerçek={dict(gt)}  TP={tp} FP={fp} FN={fn}")

    (ROOT / "results" / "real_photos.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if panels:
        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
        axes = [axes] if n == 1 else axes
        for ax, (name, im) in zip(axes, panels):
            ax.imshow(im); ax.set_title(name); ax.axis("off")
        plt.tight_layout()
        plt.savefig(ROOT / "results" / "real_photos_grid.png", dpi=130)
        print("Kaydedildi: results/real_photos_grid.png")
