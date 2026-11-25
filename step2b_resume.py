"""
ADIM 2b - EĞİTİME DEVAM (resume)

İlk eğitim 19/30 epoch'ta RAM yetersizliği nedeniyle durdu. Bu betik
runs/yolov8s_voc/weights/last.pt üzerinden kalan epoch'ları tamamlar.
Optimizer, LR planı ve epoch sayacı checkpoint'ten aynen geri yüklenir;
yalnızca bellek kullanımını azaltmak için dataloader worker sayısı 4 -> 2.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
LAST = ROOT / "runs" / "yolov8s_voc" / "weights" / "last.pt"

if __name__ == "__main__":
    model = YOLO(str(LAST))
    model.train(resume=True, workers=2)
