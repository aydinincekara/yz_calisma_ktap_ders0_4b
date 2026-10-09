# Ders 4.2: Aktarım öğrenmesi (transfer learning) ile atölye malzemesi tanıma
# Hazır model: ResNet18 (ImageNet'te 1000 sınıf için eğitilmiş)

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

torch.manual_seed(42)
cihaz = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanılan cihaz:", cihaz)

VERI_KLASORU = Path("atolye_verim")
SINIFLAR = ["arduino_uno", "breadboard", "ultrasonik_sensor", "servo_motor", "hicbiri"]
KISA_AD = ["Arduino", "Breadboard", "Ultrasonik", "Servo", "Hiçbiri"]


# 1) VERİYİ BÖL: art arda çekilen fotoğraflar birbirine çok benzer.
#    Rastgele değil, 10'luk bloklar hâlinde bölüyoruz (veri sızıntısını önlemek için).
egitim_ornek, dogrulama_ornek = [], []
for etiket, sinif in enumerate(SINIFLAR):
    dosyalar = sorted((VERI_KLASORU / sinif).glob("*.jpg"))   # dosya adı = çekim zamanı
    for i, d in enumerate(dosyalar):
        if (i // 10) % 5 == 4:          # her 5 bloktan 1'i doğrulamaya
            dogrulama_ornek.append((d, etiket))
        else:
            egitim_ornek.append((d, etiket))
print(f"Eğitim: {len(egitim_ornek)} fotoğraf, doğrulama: {len(dogrulama_ornek)} fotoğraf")


# 2) ÖN İŞLEME: hazır model ImageNet'te 224x224 ve bu renk ayarlarıyla eğitildi
ORTALAMA, SAPMA = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
artirma = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(ORTALAMA, SAPMA),
])
duz = transforms.Compose([transforms.ToTensor(), transforms.Normalize(ORTALAMA, SAPMA)])


class FotoVeri(Dataset):
    def __init__(self, ornekler, donusum):
        # Fotoğrafları bir kez okuyup 224x224'e küçültüyoruz, eğitim hızlansın
        self.goruntuler = [Image.open(yol).convert("RGB").resize((224, 224))
                           for yol, _ in ornekler]
        self.etiketler = [e for _, e in ornekler]
        self.donusum = donusum

    def __len__(self):
        return len(self.goruntuler)

    def __getitem__(self, i):
        return self.donusum(self.goruntuler[i]), self.etiketler[i]


print("Fotoğraflar yükleniyor...")
egitim_yukleyici = DataLoader(FotoVeri(egitim_ornek, artirma), batch_size=32, shuffle=True)
dogrulama_seti = FotoVeri(dogrulama_ornek, duz)
dogrulama_yukleyici = DataLoader(dogrulama_seti, batch_size=64)


def model_olustur(hazir_bilgi):
    agirliklar = ResNet18_Weights.DEFAULT if hazir_bilgi else None
    model = resnet18(weights=agirliklar)
    model.fc = nn.Linear(model.fc.in_features, len(SINIFLAR))   # 1000 sınıf yerine 5 sınıf
    return model.to(cihaz)


def dogruluk(model):
    model.eval()
    dogru = 0
    with torch.no_grad():
        for x, y in dogrulama_yukleyici:
            x, y = x.to(cihaz), y.to(cihaz)
            dogru += (model(x).argmax(1) == y).sum().item()
    return 100 * dogru / len(dogrulama_seti)


def egit(model, tur_sayisi, ogrenme_hizi, baslik):
    kayip_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=ogrenme_hizi)
    for tur in range(tur_sayisi):
        model.train()
        for x, y in egitim_yukleyici:
            x, y = x.to(cihaz), y.to(cihaz)
            kayip = kayip_fn(model(x), y)
            optimizer.zero_grad()
            kayip.backward()
            optimizer.step()
        print(f"  {baslik} - tur {tur + 1}: doğrulama doğruluğu=%{dogruluk(model):.1f}")


# 3) AŞAMA A: hazır modelin gövdesini dondur, sadece yeni son katmanı eğit
print("\nAşama A: Hazır model (ImageNet bilgisiyle), gövde donduruldu")
model = model_olustur(hazir_bilgi=True)
for ad, p in model.named_parameters():
    p.requires_grad = ad.startswith("fc.")
print(f"  Eğitimden önce: doğrulama doğruluğu=%{dogruluk(model):.1f}")
egit(model, 5, 1e-3, "Aşama A")

# 4) AŞAMA B: tüm katmanları çöz, çok küçük adımlarla ince ayar yap
print("\nAşama B: Tüm katmanlar çözüldü, ince ayar")
for p in model.parameters():
    p.requires_grad = True
egit(model, 5, 1e-4, "Aşama B")
sonuc_aktarim = dogruluk(model)

torch.save({"siniflar": SINIFLAR, "agirliklar": model.state_dict()}, "atolye_modeli.pt")
print("Model kaydedildi: atolye_modeli.pt")


# 5) KARIŞIKLIK ANALİZİ: model hangi malzemeleri birbirine karıştırdı?
print("\nHangi malzemeler birbirine karıştırıldı?")
karisiklik = {}
model.eval()
with torch.no_grad():
    for x, y in dogrulama_yukleyici:
        tahminler = model(x.to(cihaz)).argmax(1).cpu()
        for gercek, tahmin in zip(y.tolist(), tahminler.tolist()):
            if gercek != tahmin:
                cift = (KISA_AD[gercek], KISA_AD[tahmin])
                karisiklik[cift] = karisiklik.get(cift, 0) + 1
if karisiklik:
    for (gercek, tahmin), adet in sorted(karisiklik.items(), key=lambda k: -k[1]):
        print(f"  {gercek} -> {tahmin} sandı: {adet} kez")
else:
    print("  Hiç karışıklık yok!")


# 6) DENEY: aynı model, ImageNet bilgisi OLMADAN, sıfırdan eğitilirse ne olur?
print("\nDeney: Aynı model sıfırdan eğitiliyor (hazır bilgi YOK), 10 tur")
sifirdan = model_olustur(hazir_bilgi=False)
egit(sifirdan, 10, 1e-3, "Sıfırdan")
sonuc_sifirdan = dogruluk(sifirdan)

print("\n=== KARŞILAŞTIRMA (doğrulama fotoğrafları, model eğitimde görmedi) ===")
print(f"Aktarım öğrenmesi (hazır bilgi + senin fotoğrafların): %{sonuc_aktarim:.1f}")
print(f"Sıfırdan eğitim (sadece senin fotoğrafların):          %{sonuc_sifirdan:.1f}")
