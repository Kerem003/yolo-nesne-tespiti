"""
ADIM 6 - PROJE RAPORU (.docx)
results/ altındaki JSON ve grafiklerden ödev şablonuna uygun rapor üretir.
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

ROOT = Path(__file__).parent
RES = ROOT / "results"
OUT = ROOT / "Nesne_Tespiti_Proje_Raporu.docx"

train = json.loads((RES / "train_info.json").read_text(encoding="utf-8"))
test = json.loads((RES / "test_metrics.json").read_text(encoding="utf-8"))
real = json.loads((RES / "real_photos.json").read_text(encoding="utf-8")) if (RES / "real_photos.json").exists() else []
notes = json.loads((RES / "real_photos_notes.json").read_text(encoding="utf-8")) if (RES / "real_photos_notes.json").exists() else {}
hp = train["hyperparameters"]
sp = test["speed_ms"]
REPO = notes.get("repository", "")
EP = train.get("epochs_completed", hp["epochs"])
LABEL = notes.get("labels", {})

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)


def table(rows, header=None, widths=None):
    t = doc.add_table(rows=0, cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    if header:
        cells = t.add_row().cells
        for c, h in zip(cells, header):
            c.text = h; c.paragraphs[0].runs[0].bold = True
    for row in rows:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = str(v)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    doc.add_paragraph()
    return t


def image(path, width=16, caption=None):
    if Path(path).exists():
        doc.add_picture(str(path), width=Cm(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            p = doc.add_paragraph(caption); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.runs[0].italic = True; p.runs[0].font.size = Pt(9)


# ---------------- Başlık ----------------
h = doc.add_heading("COMPUTER VISION ÖDEVİ – Derin Öğrenme Uygulaması", 0)
doc.add_paragraph("YOLOv8s ile Pascal VOC Üzerinde Nesne Tespiti").runs[0].bold = True

# ---------------- 9. Rapor şablonu ----------------
doc.add_heading("Proje Özeti (Rapor Şablonu)", 1)
summary = [
    ("Project Title", "YOLOv8s ile Pascal VOC Üzerinde Gerçek Zamanlı Nesne Tespiti"),
    ("Problem / Task", "Nesne tespiti (Object Detection): görüntüdeki nesnelerin sınırlayıcı kutu (bounding box) ile konumunu ve sınıfını bulmak"),
    ("Dataset", "Pascal VOC 2007 + 2012 (20 sınıf). Train: VOC2012 trainval + VOC2007 train (14 041 görüntü), "
                "Validation: VOC2007 val (2 510), Test: VOC2007 test (4 952)"),
    ("Dataset URL", "http://host.robots.ox.ac.uk/pascal/VOC/"),
    ("Model Architecture", "YOLOv8s (Ultralytics) – CSPDarknet (C2f) backbone + SPPF + PAN-FPN neck + anchor-free decoupled detection head, "
                           "~11.1 M parametre; COCO ön-eğitimli ağırlıklarla transfer learning"),
    ("Input Size", f"{hp['imgsz']}×{hp['imgsz']}×3 (letterbox ile oran korunarak yeniden boyutlandırma)"),
    ("Hyperparameters", f"epochs={hp['epochs']} (planlanan; {EP} tamamlandı), batch={hp['batch']}, lr0={hp['lr0']}, lrf={hp['lrf']} (cosine), "
                        f"weight_decay={hp['weight_decay']}, warmup_epochs={hp['warmup_epochs']}, patience={hp['patience']}, AMP=açık"),
    ("Optimizer", f"AdamW (β1={hp['momentum']}, β2=0.999)"),
    ("Activation / Output Function", "Gizli katmanlar: SiLU. Çıkış: sınıf skorları için Sigmoid (her sınıf bağımsız), "
                                     "kutu için DFL (Distribution Focal Loss) dağılımı üzerinden Softmax → koordinat; ardından NMS"),
    ("Loss Function", "Toplam kayıp = 7.5·CIoU loss (kutu) + 0.5·BCE (sınıf) + 1.5·DFL (kutu dağılımı)"),
    ("Evaluation Metrics", "mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, ortalama IoU, sınıf bazında AP, eğitim ve tahmin süresi"),
    ("Training Environment", f"{train['gpu']} (6 GB), Windows 11, Python 3.12, PyTorch {train['torch']}, Ultralytics YOLOv8"),
    ("Best Performance", f"Test (VOC2007): mAP@0.5 = {test['mAP50']:.3f}, mAP@0.5:0.95 = {test['mAP50_95']:.3f}, "
                         f"Precision = {test['precision']:.3f}, Recall = {test['recall']:.3f}, F1 = {test['f1']:.3f}, "
                         f"ort. IoU = {test['mean_iou_matched']:.3f}"),
    ("Inference Time", f"{sp['inference']:.1f} ms/görüntü (yalnız model), uçtan uca {sp['end_to_end_wall']:.1f} ms "
                       f"(≈{test['fps_end_to_end']:.0f} FPS) – {test['gpu']}, batch=1, 640 px"),
    ("Deployment Technology", "Gradio web arayüzü (step4_app.py) – görüntü yükleme / webcam, güven ve NMS eşiği ayarı, kutulu çıktı + tespit tablosu"),
    ("Repository", REPO or "(GitHub bağlantısı eklenecek)"),
]
table(summary, header=["Alan", "Açıklama / Öğrenci yanıtı"], widths=[4.5, 12])

# ---------------- 1-2 ----------------
doc.add_heading("1. Amaç ve Görev", 1)
doc.add_paragraph(
    "Bu çalışmada bilgisayarlı görü problemlerinden nesne tespiti seçilmiştir. Amaç; bir görüntüdeki "
    "nesnelerin hem sınıfını hem de konumunu (sınırlayıcı kutu) tahmin eden bir derin öğrenme modeli "
    "geliştirmek, başarımını standart metriklerle ölçmek ve kullanıcıların kendi fotoğraflarıyla "
    "deneyebileceği bir web uygulaması olarak sunmaktır.")

# ---------------- 3 ----------------
doc.add_heading("2. Model Mimarisi", 1)
doc.add_paragraph(
    "Tek aşamalı (one-stage) bir dedektör olan YOLOv8s seçilmiştir. Faster R-CNN gibi iki aşamalı "
    "modellere göre çok daha hızlıdır ve gerçek zamanlı çalışabilir; bu da hem 6 GB'lık dizüstü GPU'da "
    "makul sürede eğitim hem de canlı uygulama için uygundur.")
for b in [
    "Backbone (CSPDarknet, C2f blokları): görüntüden çok ölçekli öznitelik haritaları çıkarır.",
    "SPPF: farklı alıcı alan (receptive field) boyutlarındaki bilgiyi birleştirir.",
    "Neck (PAN-FPN): P3/P4/P5 (stride 8/16/32) seviyelerini yukarıdan-aşağı ve aşağıdan-yukarı birleştirir; küçük ve büyük nesneler için.",
    "Head (anchor-free, decoupled): her ölçekte sınıflandırma ve kutu regresyonunu ayrı dallarda yapar; önceden tanımlı anchor kutusu gerekmez.",
    "Transfer learning: COCO (80 sınıf) ağırlıklarıyla başlatıldı, çıkış katmanı 20 VOC sınıfına göre yeniden oluşturuldu ve tüm ağ ince ayarlandı.",
]:
    doc.add_paragraph(b, style="List Bullet")

# ---------------- 4 ----------------
doc.add_heading("3. Veri Seti", 1)
doc.add_paragraph(
    "Pascal VOC, nesne tespiti için yaygın kullanılan açık kaynaklı bir veri setidir. 20 sınıf içerir: "
    "aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, "
    "person, pottedplant, sheep, sofa, train, tvmonitor. XML etiketleri YOLO formatına "
    "(sınıf, x_merkez, y_merkez, genişlik, yükseklik – normalize) dönüştürülmüş, 'difficult' işaretli nesneler çıkarılmıştır.")
table([("Eğitim (train)", "VOC2012 trainval + VOC2007 train", "14 041"),
       ("Doğrulama (val)", "VOC2007 val", "2 510"),
       ("Test", "VOC2007 test", "4 952")],
      header=["Küme", "Kaynak", "Görüntü sayısı"])
doc.add_paragraph("Doğrulama kümesi eğitim sırasında en iyi modelin (best.pt) seçimi ve erken durdurma için; "
                  "test kümesi ise yalnızca nihai değerlendirmede kullanılmıştır.")

# ---------------- 5 ----------------
doc.add_heading("4. Model Ayarları ve Eğitim", 1)
table([
    ("Input size", f"{hp['imgsz']}", "YOLOv8'in ön-eğitildiği çözünürlük; küçük nesneler için yeterli detay."),
    ("Epoch", f"{hp['epochs']} planlandı, {EP} tamamlandı (patience {hp['patience']})", "COCO ağırlıkları VOC sınıflarını zaten büyük ölçüde içerdiği için hızlı yakınsar."),
    ("Batch size", f"{hp['batch']}", "6 GB VRAM'e AMP ile sığan en büyük makul değer."),
    ("Optimizer", "AdamW", "Adaptif öğrenme oranı + ayrık (decoupled) weight decay; ince ayarda kararlı."),
    ("Learning rate", f"{hp['lr0']} → {hp['lr0']*hp['lrf']:.0e} (cosine)", "Ön-eğitimli ağırlıkları bozmamak için düşük başlangıç, kosinüs ile yumuşak azalma."),
    ("Warmup", f"{hp['warmup_epochs']} epoch", "İlk adımlarda yeni çıkış katmanının büyük gradyanlarını sönümler."),
    ("Weight decay", f"{hp['weight_decay']}", "L2 düzenlileştirme, aşırı öğrenmeyi azaltır."),
    ("Veri artırma", "Mosaic, HSV, yatay çevirme, ölçek/öteleme", "Ultralytics varsayılanları; son 10 epoch'ta mosaic kapatılır."),
    ("Aktivasyon / çıkış", "SiLU / Sigmoid + DFL", "Sigmoid çoklu-etiket sınıf skoru, DFL kutu kenarlarını olasılık dağılımı olarak öğrenir."),
    ("Loss", "CIoU + BCE + DFL", "CIoU: örtüşme + merkez uzaklığı + en-boy oranı; BCE: sınıf; DFL: hassas lokalizasyon."),
], header=["Parametre", "Değer", "Gerekçe"], widths=[3.3, 4, 9.2])
doc.add_paragraph(f"Toplam eğitim süresi: {train['train_time_min']:.1f} dakika ({train['gpu']}).")
doc.add_paragraph(
    f"Not: Eğitim {hp['epochs']} epoch olarak planlanmıştır. 19. epoch'tan sonra sistem belleği (RAM) yetersiz kaldığı için "
    "süreç durmuş, kaydedilen son ağırlıklardan (last.pt) resume ile devam edilmiş ve veri yükleyici işçi sayısı 2'ye düşürülmüştür. "
    f"30. epoch sırasında süreç tekrar kesildiğinden nihai model {EP}. epoch ağırlıklarıdır. Şekil 1'de görüldüğü gibi doğrulama "
    "mAP değerleri son epoch'larda neredeyse sabitlenmiştir (28. epoch mAP@0.5 = 0.832, 29. epoch 0.833); bu nedenle eksik kalan "
    "tek epoch'un sonuca etkisi ihmal edilebilir düzeydedir.")
image(RES / "training_curves.png", 16.5, "Şekil 1 – Eğitim/doğrulama kayıpları ve doğrulama metrikleri")

# ---------------- 6 ----------------
doc.add_heading("5. Performans Değerlendirmesi (Test kümesi)", 1)
table([
    ("mAP@0.5", f"{test['mAP50']:.4f}"),
    ("mAP@0.5:0.95", f"{test['mAP50_95']:.4f}"),
    ("Precision", f"{test['precision']:.4f}"),
    ("Recall", f"{test['recall']:.4f}"),
    ("F1-score", f"{test['f1']:.4f}"),
    ("Ortalama IoU (eşleşen kutular, IoU>0.5)", f"{test['mean_iou_matched']:.4f}"),
    ("Gerçek kutuların bulunma oranı (IoU>0.5, conf≥0.25)", f"{test['gt_match_rate_iou50']:.4f}"),
    ("Tahmin süresi – ön işleme", f"{sp['preprocess']:.2f} ms"),
    ("Tahmin süresi – model (inference)", f"{sp['inference']:.2f} ms"),
    ("Tahmin süresi – son işleme (NMS)", f"{sp['postprocess']:.2f} ms"),
    ("Uçtan uca tahmin süresi / FPS", f"{sp['end_to_end_wall']:.2f} ms / {test['fps_end_to_end']:.0f} FPS"),
    ("Eğitim süresi", f"{train['train_time_min']:.1f} dk"),
], header=["Metrik", "Değer"])
doc.add_paragraph(
    "IoU (Intersection over Union), tahmin kutusu ile gerçek kutunun kesişim alanının birleşim alanına oranıdır. "
    "Bir tahmin, IoU ≥ 0.5 ise doğru (TP) sayılır. AP, bir sınıfın Precision-Recall eğrisinin altındaki alandır; "
    "mAP@0.5 tüm sınıfların AP ortalamasıdır. mAP@0.5:0.95 ise 0.50'den 0.95'e kadar 10 IoU eşiğinin ortalamasıdır "
    "ve lokalizasyon hassasiyetini de ölçtüğü için daha zorlayıcıdır.")
image(RES / "per_class_ap50.png", 12, "Şekil 2 – Sınıf bazında AP@0.5")
pc = sorted(test["per_class_ap50"].items(), key=lambda x: -x[1])
doc.add_paragraph(
    f"En başarılı sınıflar: {', '.join(f'{k} ({v:.2f})' for k, v in pc[:3])}. "
    f"En zor sınıflar: {', '.join(f'{k} ({v:.2f})' for k, v in pc[-3:])}. "
    "Şişe, saksı bitkisi ve sandalye gibi sınıflar genellikle küçük, kalabalık ortamda ve birbirini örten (occlusion) "
    "nesneler olduğundan daha düşük AP alır; büyük ve belirgin şekilli nesneler (kedi, tren, uçak) ise daha kolay tespit edilir.")
for f, cap in [("confusion_matrix_normalized.png", "Şekil 3 – Normalize karışıklık matrisi (test)"),
               ("BoxPR_curve.png", "Şekil 4 – Precision-Recall eğrileri (test)")]:
    image(ROOT / "runs" / "test_eval" / f, 14, cap)

# ---------------- 7 ----------------
doc.add_heading("6. Uygulama (Deployment)", 1)
doc.add_paragraph(
    "Eğitilen model Gradio ile web uygulamasına dönüştürülmüştür (step4_app.py). Kullanıcı bilgisayarından "
    "görüntü yükleyebilir, panodan yapıştırabilir veya web kamerasıyla fotoğraf çekebilir. Güven eşiği ve "
    "NMS IoU eşiği kaydırıcılarla ayarlanabilir. Çıktı olarak kutular çizilmiş görüntü, tespit sayısı, tahmin "
    "süresi ve her tespitin sınıfı (İngilizce/Türkçe), güven skoru ve koordinatlarını içeren tablo gösterilir.")
doc.add_paragraph("Çalıştırma: .venv\\Scripts\\python.exe step4_app.py  →  http://127.0.0.1:7860")
image(RES / "gradio_screenshot.png", 16, "Şekil 5a – Gradio arayüzü: sandalye fotoğrafı (solda girdi, sağda model çıktısı)")
image(RES / "gradio_screenshot2.png", 16, "Şekil 5b – Gradio arayüzü: şişe fotoğrafı")
doc.add_paragraph("Ekran görüntülerinde arayüzün üst kısmı yer almaktadır; sayfanın alt kısmında güven/NMS eşiği "
                  "kaydırıcıları, 'Tespit et' butonu, tespit sayısı ve süre bilgisi ile tespit tablosu bulunur.")

# ---------------- 8 ----------------
doc.add_heading("7. Gerçek Görüntülerle Test", 1)
if real:
    doc.add_paragraph("Telefonla çekilen 5 fotoğraf modele verilmiş (güven eşiği 0.25), tahminler gerçek nesne "
                      "sayılarıyla karşılaştırılmıştır.")
    image(RES / "real_photos_grid.png", 17, "Şekil 6 – Kendi fotoğraflarımızda model çıktıları")
    rows = []
    for r in real:
        fmt = lambda d: ", ".join(f"{k}×{v}" for k, v in d.items()) or "—"
        rows.append((LABEL.get(r["photo"], r["photo"]), fmt(r["ground_truth"]), fmt(r["predicted"]), r["TP"], r["FP"], r["FN"]))
    table(rows, header=["Fotoğraf", "Gerçek", "Tahmin", "TP", "FP", "FN"])
    doc.add_paragraph("Gerçek değer belirlenirken yalnızca kadrajda tamamen görünen VOC sınıfı nesneler sayılmıştır. "
                      "Kısmen görünen nesneler ve VOC dışı nesneler (defter, kalem, şemsiye) gerçek değer olarak alınmamıştır.")

    # Güven eşiği ödünleşimi
    doc.add_heading("Güven eşiğinin etkisi", 3)
    def score(th):
        tp = fp = fn = 0
        for r in real:
            pred = {}
            for d in r["detections"]:
                if d["conf"] >= th:
                    pred[d["class"]] = pred.get(d["class"], 0) + 1
            for c in set(pred) | set(r["ground_truth"]):
                p, g = pred.get(c, 0), r["ground_truth"].get(c, 0)
                tp += min(p, g); fp += max(p - g, 0); fn += max(g - p, 0)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        return (f"{th:.2f}", tp, fp, fn, f"{prec:.3f}", f"{rec:.3f}")
    table([score(t) for t in (0.25, 0.4, 0.6)], header=["Güven eşiği", "TP", "FP", "FN", "Precision", "Recall"])
    doc.add_paragraph(
        "Eşik yükseltildikçe düşük güvenli yanlış tespitler (defter → tvmonitor 0.58, kalem → aeroplane 0.45, arka plandaki "
        "kısmi sandalyeler) elenir ve Precision artar. Hedef nesneler (sandalye 0.85, şişe 0.87) yüksek güvenle bulunduğu için "
        "bu fotoğraflarda Recall düşmez. Ancak genel kullanımda yüksek eşik, küçük veya örtülü nesnelerin kaçırılmasına (FN) yol açar. "
        "Bu nedenle uygulamada eşik kullanıcıya bırakılmış, varsayılan değer 0.25 seçilmiştir: tüm olası nesneleri göstermek için "
        "düşük, yalnızca emin olunan tespitleri görmek için 0.5–0.6 aralığı önerilir.")

    for r in real:
        img = RES / "real_photos" / f"{Path(r['photo']).stem}_pred.jpg"
        doc.add_heading(LABEL.get(r["photo"], r["photo"]), 3)
        image(img, 11)
        if r["photo"] in notes.get("comments", {}):
            doc.add_paragraph(notes["comments"][r["photo"]])
    if notes.get("general"):
        doc.add_heading("Genel değerlendirme", 3)
        doc.add_paragraph(notes["general"])
else:
    doc.add_paragraph("(Telefon fotoğrafları eklendikten sonra step5_real_photos.py çalıştırılıp rapor yeniden üretilecek.)")

# ---------------- Sonuç ----------------
doc.add_heading("8. Sonuç", 1)
doc.add_paragraph(
    f"COCO ön-eğitimli YOLOv8s, Pascal VOC üzerinde {EP} epoch ince ayar ile test kümesinde "
    f"mAP@0.5 = {test['mAP50']:.3f} ve mAP@0.5:0.95 = {test['mAP50_95']:.3f} başarıma ulaşmış; dizüstü GPU'da "
    f"görüntü başına ~{sp['end_to_end_wall']:.0f} ms ile gerçek zamanlı çalışabilmektedir. Model büyük ve belirgin "
    "nesnelerde çok başarılıyken küçük, örtülü ve kalabalık sahnelerdeki nesnelerde zorlanmaktadır. Daha büyük bir model "
    "(YOLOv8m/l), daha yüksek çözünürlük veya daha uzun eğitim ile başarım artırılabilir; ayrıca model yalnızca 20 VOC "
    "sınıfını tanıdığından bu sınıflar dışındaki nesneler (ör. telefon, bardak, kitap) tespit edilemez.")

doc.add_heading("Dosya Yapısı", 1)
for f in ["step1_data.py – veri indirme, YOLO formatına dönüştürme, train/val/test ayırma",
          "step2_train.py – YOLOv8s eğitimi", "step3_evaluate.py – test metrikleri, IoU, süre ölçümü, grafikler",
          "step4_app.py – Gradio uygulaması", "step5_real_photos.py – telefon fotoğraflarıyla test",
          "step6_report.py – bu raporun üretimi", "runs/yolov8s_voc/weights/best.pt – eğitilmiş model"]:
    doc.add_paragraph(f, style="List Bullet")

doc.save(OUT)
print("Rapor kaydedildi:", OUT)
