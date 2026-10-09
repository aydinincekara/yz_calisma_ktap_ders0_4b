# Ders 4.3: Canlı kamera ile atölye malzemesi tanıma
# Not: atolye_modeli.pt (Ders 4.2'nin çıktısı) bu dosyayla aynı klasörde olmalı.
# Çıkmak için kamera penceresi seçiliyken Q tuşuna bas.

from collections import deque

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image, ImageDraw, ImageFont

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
ESIK = 0.70             # bu değerin altında "Emin değilim"
YUMUSATMA = 5           # son kaç karenin ortalaması alınacak
PENCERE = "Atolye Malzemesi Tanima - cikmak icin Q"

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

# Ön işleme: Ders 4.2'deki doğrulama ön işlemesinin aynısı
on_isleme = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Türkçe karakterler için Windows'un Arial yazı tipini kullan
try:
    buyuk_font = ImageFont.truetype("arial.ttf", 32)
    kucuk_font = ImageFont.truetype("arial.ttf", 20)
except OSError:
    buyuk_font = kucuk_font = ImageFont.load_default()

kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not kamera.isOpened():
    print("Kamera açılamadı. Kamerayı kullanan başka bir program açık olabilir.")
    raise SystemExit

son_olasiliklar = deque(maxlen=YUMUSATMA)
print("Kamera açıldı. Çıkmak için kamera penceresi seçiliyken Q tuşuna bas.")

while True:
    okundu, kare = kamera.read()
    if not okundu:
        print("Kameradan görüntü alınamadı.")
        break

    # OpenCV renkleri BGR sırasıyla okur, model RGB bekler
    goruntu = Image.fromarray(cv2.cvtColor(kare, cv2.COLOR_BGR2RGB))
    x = on_isleme(goruntu.resize((224, 224))).unsqueeze(0).to(cihaz)
    with torch.no_grad():
        olasiliklar = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    # Zamansal yumuşatma: son birkaç karenin ortalaması
    son_olasiliklar.append(olasiliklar)
    ortalama = np.mean(son_olasiliklar, axis=0)
    en_iyi = int(ortalama.argmax())
    eminlik = ortalama[en_iyi]

    if eminlik < ESIK:
        yazi, renk = f"Emin değilim (%{eminlik * 100:.0f})", (255, 200, 0)
    elif siniflar[en_iyi] == "hicbiri":
        yazi, renk = "Malzeme görmüyorum", (200, 200, 200)
    else:
        yazi, renk = f"{GORUNEN_AD[siniflar[en_iyi]]} (%{eminlik * 100:.0f})", (0, 255, 0)

    # Sonucu görüntünün üzerine yaz
    ciz = ImageDraw.Draw(goruntu)
    ciz.rectangle([0, 0, goruntu.width, 50], fill=(0, 0, 0))
    ciz.text((10, 8), yazi, font=buyuk_font, fill=renk)

    y = goruntu.height - 25 * len(siniflar) - 10
    ciz.rectangle([0, y - 5, 270, goruntu.height], fill=(0, 0, 0))
    for i, sinif in enumerate(siniflar):
        ciz.text((10, y + 25 * i), f"{GORUNEN_AD[sinif]}: %{ortalama[i] * 100:.0f}",
                 font=kucuk_font, fill=(255, 255, 255))

    cv2.imshow(PENCERE, cv2.cvtColor(np.array(goruntu), cv2.COLOR_RGB2BGR))

    tus = cv2.waitKey(1) & 0xFF
    if tus in (ord("q"), ord("Q")):
        break
    if cv2.getWindowProperty(PENCERE, cv2.WND_PROP_VISIBLE) < 1:   # pencere X ile kapatıldıysa
        break

kamera.release()
cv2.destroyAllWindows()
