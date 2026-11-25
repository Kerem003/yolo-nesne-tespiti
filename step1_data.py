"""
ADIM 1 - VERİ SETİ HAZIRLAMA (Pascal VOC 2007 + 2012)

- VOC 2007 ve VOC 2012 arşivlerini indirir.
- XML (Pascal VOC formatı) etiketlerini YOLO formatına çevirir:
      <sınıf_id> <x_merkez> <y_merkez> <genişlik> <yükseklik>   (0-1 arası normalize)
- Eğitim / Doğrulama / Test kümelerini AYRI klasörlere koyar:
      Train : VOC2012 trainval + VOC2007 train   (~14k görüntü)
      Val   : VOC2007 val                        (~2.5k görüntü)
      Test  : VOC2007 test                       (~4.9k görüntü, eğitimde HİÇ kullanılmaz)
- voc.yaml veri tanım dosyasını oluşturur.
"""
import os
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "datasets" / "VOC"
RAW = DATA / "raw"

URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtest_06-Nov-2007.zip",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_11-May-2012.zip",
]

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# (yıl, VOC image-set, hedef küme)
SPLITS = [
    ("2012", "trainval", "train"),
    ("2007", "train", "train"),
    ("2007", "val", "val"),
    ("2007", "test", "test"),
]


def download_and_extract():
    RAW.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        zip_path = RAW / Path(url).name
        if not zip_path.exists():
            print(f"İndiriliyor: {url}")
            part = zip_path.with_suffix(".part")  # yarıda kalan indirme tamamlanmış sanılmasın
            urllib.request.urlretrieve(url, part)
            part.replace(zip_path)
        marker = zip_path.with_suffix(".done")
        if not marker.exists():
            print(f"Açılıyor: {zip_path.name}")
            shutil.unpack_archive(zip_path, RAW)
            marker.touch()


def find_voc_dir(year):
    for p in RAW.rglob(f"VOC{year}"):
        if (p / "Annotations").is_dir():
            return p
    raise FileNotFoundError(f"VOC{year} klasörü bulunamadı")


def convert_box(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x, y = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1
    w, h = box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_annotation(xml_path, txt_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        difficult = obj.find("difficult")
        if cls not in CLASSES or (difficult is not None and int(difficult.text) == 1):
            continue
        b = obj.find("bndbox")
        bb = convert_box((w, h), [float(b.find(t).text) for t in ("xmin", "xmax", "ymin", "ymax")])
        lines.append(f"{CLASSES.index(cls)} " + " ".join(f"{a:.6f}" for a in bb))
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_yolo_splits():
    counts = {"train": 0, "val": 0, "test": 0}
    for year, image_set, split in SPLITS:
        voc = find_voc_dir(year)
        img_dir = DATA / "images" / split
        lbl_dir = DATA / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        ids = (voc / "ImageSets" / "Main" / f"{image_set}.txt").read_text().split()
        for img_id in ids:
            name = f"{year}_{img_id}"
            src = voc / "JPEGImages" / f"{img_id}.jpg"
            dst = img_dir / f"{name}.jpg"
            if not dst.exists():
                shutil.copy(src, dst)
            convert_annotation(voc / "Annotations" / f"{img_id}.xml", lbl_dir / f"{name}.txt")
        counts[split] += len(ids)
        print(f"VOC{year} {image_set:8s} -> {split:5s}: {len(ids)} görüntü")
    return counts


def write_yaml():
    yaml = [f"path: {DATA.as_posix()}", "train: images/train", "val: images/val",
            "test: images/test", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (ROOT / "voc.yaml").write_text("\n".join(yaml) + "\n")


if __name__ == "__main__":
    download_and_extract()
    counts = build_yolo_splits()
    write_yaml()
    print("Toplam:", counts)
    print("voc.yaml oluşturuldu.")
