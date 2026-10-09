# Ders 4.4: Grad-CAM ile modelin nereye baktığını görüyoruz
# Not: atolye_modeli.pt (Ders 4.2'nin çıktısı) bu dosyayla aynı klasörde olmalı.
# Tuşlar: 0 = otomatik, 1-5 = sınıf seç, S = ekranı kaydet, Q = çık
# Sürüm 2: Kayıtlar gradcam_kayitlari klasörüne, sınıf adıyla kaydedilir.

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image, ImageDraw, ImageFont

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
PENCERE = "Grad-CAM - 0: otomatik, 1-5: sinif sec, S: kaydet, Q: cik"
KAYIT_KLASORU = Path("gradcam_kayitlari")
KAYIT_KLASORU.mkdir(exist_ok=True)

GORUNEN_AD = {
    "arduino_uno": "Arduino Uno",
    "breadboard": "Breadboard",
    "ultrasonik_sensor": "Ultrasonik Sensör",
    "servo_motor": "Servo Motor",
    "hicbiri": "Hiçbiri",
}

# Modeli yükle: Ders 4.2'deki yapının aynısı
kayit = torch.load("atolye_modeli.pt", map_location=cihaz)
siniflar = kayit["siniflar"]
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(siniflar))
model.load_state_dict(kayit["agirliklar"])
model.to(cihaz).eval()

on_isleme = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# GRAD-CAM KANCASI: son evrişim katmanının (layer4) çıktısını ve gradyanını yakala
yakalanan = {}


def ileri_kanca(modul, girdi, cikti):
    yakalanan["aktivasyon"] = cikti
    cikti.register_hook(lambda gradyan: yakalanan.__setitem__("gradyan", gradyan))


model.layer4.register_forward_hook(ileri_kanca)


def grad_cam(x, hedef_sinif=None):
    """Olasılıkları ve seçilen sınıf için 7x7'lik ısı haritasını döndürür."""
    cikti = model(x)
    olasiliklar = torch.softmax(cikti, dim=1)[0].detach().cpu().numpy()
    if hedef_sinif is None:
        hedef_sinif = int(olasiliklar.argmax())

    model.zero_grad(set_to_none=True)
    cikti[0, hedef_sinif].backward()      # Ders 1'deki geri yayılım, bu sefer ölçmek için

    aktivasyon = yakalanan["aktivasyon"][0]          # (512, 7, 7): 512 özellik haritası
    gradyan = yakalanan["gradyan"][0]                # her haritanın karara etkisi
    agirlik = gradyan.mean(dim=(1, 2))               # her özellik haritasının önemi
    harita = torch.relu((agirlik[:, None, None] * aktivasyon).sum(dim=0))
    harita = harita / (harita.max() + 1e-8)
    return olasiliklar, hedef_sinif, harita.detach().cpu().numpy()


try:
    font = ImageFont.truetype("arial.ttf", 24)
except OSError:
    font = ImageFont.load_default()


def yazi_ekle(bgr, satirlar):
    """Türkçe karakterleri doğru göstermek için yazıyı PIL ile çizer."""
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ciz = ImageDraw.Draw(pil)
    ciz.rectangle([0, 0, pil.width, 32 * len(satirlar) + 8], fill=(0, 0, 0))
    for i, (metin, renk) in enumerate(satirlar):
        ciz.text((10, 6 + 32 * i), metin, font=font, fill=renk)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not kamera.isOpened():
    print("Kamera açılamadı. Kamerayı kullanan başka bir program açık olabilir.")
    raise SystemExit

print("Kamera açıldı.")
print("Tuşlar: 0 = otomatik | 1 = Arduino | 2 = Breadboard | 3 = Ultrasonik | "
      "4 = Servo | 5 = Hiçbiri | S = kaydet | Q = çık")
print(f"Kayıtlar şu klasöre gidecek: {KAYIT_KLASORU.resolve()}")

secili_sinif = None        # None = otomatik (modelin seçtiği sınıf)
onceki_harita = None

while True:
    okundu, kare = kamera.read()
    if not okundu:
        print("Kameradan görüntü alınamadı.")
        break

    rgb = Image.fromarray(cv2.cvtColor(kare, cv2.COLOR_BGR2RGB))
    x = on_isleme(rgb.resize((224, 224))).unsqueeze(0).to(cihaz)
    olasiliklar, gosterilen, harita = grad_cam(x, secili_sinif)

    # Titremeyi azaltmak için ısı haritasını bir önceki kareyle harmanla
    if onceki_harita is not None:
        harita = 0.6 * onceki_harita + 0.4 * harita
    onceki_harita = harita

    # 7x7 haritayı kamera görüntüsü boyutuna büyüt ve renklendir
    h, w = kare.shape[:2]
    buyuk_harita = cv2.resize(harita, (w, h))
    renkli = cv2.applyColorMap((buyuk_harita * 255).astype(np.uint8), cv2.COLORMAP_JET)
    karisim = cv2.addWeighted(kare, 0.5, renkli, 0.5, 0)

    en_iyi = int(olasiliklar.argmax())
    mod = "otomatik" if secili_sinif is None else "elle seçildi"
    sol = yazi_ekle(kare.copy(), [
        (f"Modelin kararı: {GORUNEN_AD[siniflar[en_iyi]]} (%{olasiliklar[en_iyi] * 100:.0f})",
         (0, 255, 0)),
    ])
    sag = yazi_ekle(karisim, [
        (f"Isı haritası: {GORUNEN_AD[siniflar[gosterilen]]} ({mod})", (255, 255, 0)),
        (f"Bu sınıfın olasılığı: %{olasiliklar[gosterilen] * 100:.0f}", (255, 255, 255)),
    ])
    birlesik = np.hstack([sol, sag])
    cv2.imshow(PENCERE, birlesik)

    tus = cv2.waitKey(1) & 0xFF
    if tus in (ord("q"), ord("Q")):
        break
    if tus == ord("0"):
        secili_sinif = None
    elif ord("1") <= tus <= ord(str(len(siniflar))):
        secili_sinif = tus - ord("1")
    elif tus in (ord("s"), ord("S")):
        zaman = time.strftime("%Y%m%d_%H%M%S")
        dosya = KAYIT_KLASORU / f"{zaman}_{siniflar[gosterilen]}.jpg"
        cv2.imwrite(str(dosya), birlesik)
        print(f"Kaydedildi: {dosya}")
    if cv2.getWindowProperty(PENCERE, cv2.WND_PROP_VISIBLE) < 1:
        break

kamera.release()
cv2.destroyAllWindows()
