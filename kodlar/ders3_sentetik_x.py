# Ders 3.2c: Sentetik "rakam değil" (X) verisi — düzeltilmiş sürüm
# Düzeltmeler: ince harfler, kendi üzerinden geçmeyen zikzaklar, elle çizilmiş gibi kalpler
# Harf dosyalarının adında harfin kendisi yazar (örnek: harf_0050_R.png)

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

random.seed(7)
HEDEF = Path("sentetik_X")
HEDEF.mkdir(exist_ok=True)

# Önceki üretimden kalan görselleri temizle
eskiler = list(HEDEF.glob("*.png"))
for dosya in eskiler:
    dosya.unlink()
print(f"Eski sentetik görseller silindi: {len(eskiler)}")

# Rakama benzeyen harfler bilerek çıkarıldı (O-0, I-1, Z-2, S-5, G-6, B-8, g-9, T-7 gibi)
HARFLER = "ACDEFHKLMNPRUVWXacdefhkmnpruvwxÇçÜü"
FONTLAR = ["arial.ttf", "times.ttf", "comic.ttf", "georgia.ttf", "verdana.ttf"]


def ortala_28(img):
    """Ders 2'deki biçimin aynısı: kırp, 20x20 kutuya sığdır, 28x28 ortasına koy."""
    a = np.array(img)
    satirlar = np.where(a.max(axis=1) > 30)[0]
    sutunlar = np.where(a.max(axis=0) > 30)[0]
    if len(satirlar) == 0:
        return None
    kirpik = a[satirlar[0]:satirlar[-1] + 1, sutunlar[0]:sutunlar[-1] + 1]
    h, w = kirpik.shape
    olcek = 20 / max(h, w)
    kucuk = Image.fromarray(kirpik).resize(
        (max(1, round(w * olcek)), max(1, round(h * olcek))), Image.LANCZOS)
    tuval = Image.new("L", (28, 28), 0)
    tuval.paste(kucuk, ((28 - kucuk.width) // 2, (28 - kucuk.height) // 2))
    return tuval


def harf_uret(font):
    img = Image.new("L", (240, 240), 0)
    d = ImageDraw.Draw(img)
    harf = random.choice(HARFLER)
    d.text((40, 20), harf, fill=255, font=font,
           stroke_width=random.randint(1, 4), stroke_fill=255)   # ince: delikler kapanmasın
    return img.rotate(random.uniform(-15, 15)), harf


def zikzak_uret():
    img = Image.new("L", (280, 280), 0)
    d = ImageDraw.Draw(img)
    n = random.randint(4, 6)
    while True:
        xs = sorted(random.sample(range(30, 251), n))   # soldan sağa sıralı: kendi üzerinden geçmez
        if xs[-1] - xs[0] > 120:                        # çok dar olursa 1'e benzer, tekrar dene
            break
    noktalar = [(x, random.randint(30, 250)) for x in xs]
    d.line(noktalar, fill=255, width=18, joint="curve")
    return img


def yildiz_uret():
    img = Image.new("L", (280, 280), 0)
    d = ImageDraw.Draw(img)
    r = random.randint(80, 120)
    bas = random.uniform(0, 2 * np.pi)
    noktalar = [(140 + r * np.cos(bas + i * 4 * np.pi / 5) + random.uniform(-12, 12),
                 140 + r * np.sin(bas + i * 4 * np.pi / 5) + random.uniform(-12, 12))
                for i in range(6)]
    d.line(noktalar, fill=255, width=18, joint="curve")
    return img


def kalp_uret():
    img = Image.new("L", (280, 280), 0)
    d = ImageDraw.Draw(img)
    t = np.linspace(0, 2 * np.pi, 60)
    olcek = random.uniform(5, 7)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    noktalar = [(140 + olcek * xi + random.uniform(-3, 3),
                 140 + olcek * yi + random.uniform(-3, 3)) for xi, yi in zip(x, y)]
    d.line(noktalar, fill=255, width=18, joint="curve")
    return img.rotate(random.uniform(-20, 20))


fontlar = []
for ad in FONTLAR:
    try:
        fontlar.append(ImageFont.truetype(ad, 140))
    except OSError:
        print(f"Yazı tipi bulunamadı, atlanıyor: {ad}")
print(f"Kullanılan yazı tipi sayısı: {len(fontlar)}")

uretim_plani = [("harf", 1000), ("zikzak", 800), ("yildiz", 500), ("kalp", 500)]
toplam = 0
for tur, adet in uretim_plani:
    if tur == "harf" and not fontlar:
        continue
    for i in range(adet):
        if tur == "harf":
            img, harf = harf_uret(random.choice(fontlar))
            dosya_adi = f"harf_{i:04d}_{harf}.png"
        elif tur == "zikzak":
            img, dosya_adi = zikzak_uret(), f"zikzak_{i:04d}.png"
        elif tur == "yildiz":
            img, dosya_adi = yildiz_uret(), f"yildiz_{i:04d}.png"
        else:
            img, dosya_adi = kalp_uret(), f"kalp_{i:04d}.png"
        kucuk = ortala_28(img)
        if kucuk is not None:
            kucuk.save(HEDEF / dosya_adi)
            toplam += 1

print(f"{toplam} sentetik X görseli üretildi -> {HEDEF}\\")
