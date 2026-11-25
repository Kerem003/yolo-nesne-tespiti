# YOLOv8s ile Pascal VOC Üzerinde Nesne Tespiti

Derin Öğrenme dersi – Computer Vision ödevi. COCO ön-eğitimli **YOLOv8s**, **Pascal VOC 2007+2012** (20 sınıf) üzerinde ince ayarlanmış ve **Gradio** ile web uygulamasına dönüştürülmüştür.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install ultralytics gradio pandas matplotlib python-docx
```

## Adımlar

| Dosya | Açıklama |
|---|---|
| `step1_data.py` | VOC'u indirir, XML → YOLO formatına çevirir, train / val / test ayırır |
| `step2_train.py` | YOLOv8s eğitimi (AdamW, 640 px, cosine LR, AMP) |
| `step3_evaluate.py` | Test kümesinde mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, IoU, tahmin süresi |
| `step4_app.py` | Gradio uygulaması → http://127.0.0.1:7860 |
| `step5_real_photos.py` | Telefonla çekilen 5 fotoğrafla test, tahmin vs gerçek karşılaştırması |
| `step6_report.py` | Proje raporunu (`Nesne_Tespiti_Proje_Raporu.docx`) üretir |

```bash
.venv\Scripts\python step1_data.py
.venv\Scripts\python step2_train.py
.venv\Scripts\python step3_evaluate.py
.venv\Scripts\python step4_app.py
```

## Veri kümeleri

| Küme | Kaynak | Görüntü |
|---|---|---|
| Train | VOC2012 trainval + VOC2007 train | 14 041 |
| Val | VOC2007 val | 2 510 |
| Test | VOC2007 test | 4 952 |

Sonuçlar `results/` klasöründe ve proje raporunda yer alır.
