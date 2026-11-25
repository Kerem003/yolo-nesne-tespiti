"""
ADIM 4 - UYGULAMA (Gradio)

Çalıştırma:  .venv\\Scripts\\python.exe step4_app.py   ->  http://127.0.0.1:7860
Kullanıcı bir fotoğraf yükler (veya kamerayla çeker), güven eşiğini seçer;
model tespit edilen nesneleri kutularla çizer ve tablo halinde listeler.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

ROOT = Path(__file__).parent
model = YOLO(str(ROOT / "runs" / "yolov8s_voc" / "weights" / "best.pt"))

TR = {"aeroplane": "uçak", "bicycle": "bisiklet", "bird": "kuş", "boat": "tekne",
      "bottle": "şişe", "bus": "otobüs", "car": "araba", "cat": "kedi", "chair": "sandalye",
      "cow": "inek", "diningtable": "yemek masası", "dog": "köpek", "horse": "at",
      "motorbike": "motosiklet", "person": "insan", "pottedplant": "saksı bitkisi",
      "sheep": "koyun", "sofa": "kanepe", "train": "tren", "tvmonitor": "TV/monitör"}


def detect(image, conf, iou):
    if image is None:
        return None, [], ""
    t0 = time.time()
    r = model.predict(image, conf=conf, iou=iou, imgsz=640, verbose=False)[0]
    ms = (time.time() - t0) * 1000
    annotated = r.plot()[:, :, ::-1]  # r.plot() BGR döndürür -> RGB
    rows = []
    for b in r.boxes:
        name = model.names[int(b.cls)]
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        rows.append([name, TR.get(name, name), round(float(b.conf), 3), f"({x1},{y1})-({x2},{y2})"])
    rows.sort(key=lambda x: -x[2])
    summary = f"**{len(rows)} nesne** tespit edildi · tahmin süresi: **{ms:.0f} ms**"
    return annotated, rows, summary


with gr.Blocks(title="YOLOv8 Nesne Tespiti") as demo:
    gr.Markdown("# YOLOv8s Nesne Tespiti (Pascal VOC, 20 sınıf)\n"
                "Bir fotoğraf yükleyin. Desteklenen sınıflar: " + ", ".join(TR.values()))
    with gr.Row():
        with gr.Column():
            # PIL: ultralytics PIL girdiyi RGB kabul edip kendisi BGR'ye çevirir (numpy verilseydi BGR sanılırdı)
            inp = gr.Image(type="pil", label="Girdi görüntüsü", sources=["upload", "webcam", "clipboard"])
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Güven eşiği (confidence)")
            iou = gr.Slider(0.1, 0.9, value=0.7, step=0.05, label="NMS IoU eşiği")
            btn = gr.Button("Tespit et", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Sonuç")
            out_txt = gr.Markdown()
            out_tbl = gr.Dataframe(headers=["Sınıf", "Türkçe", "Güven", "Kutu (x1,y1)-(x2,y2)"],
                                   label="Tespitler")
    btn.click(detect, [inp, conf, iou], [out_img, out_tbl, out_txt])
    inp.change(detect, [inp, conf, iou], [out_img, out_tbl, out_txt])

if __name__ == "__main__":
    demo.launch()
