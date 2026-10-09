# Ders 3.2: MNIST + kendi verimiz ile CNN modeli eğitiyoruz
# Sınıflar: 0-9 rakamları + "X" (rakam olmayan çizimler) = 11 sınıf

import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image

torch.manual_seed(42)
random.seed(42)
cihaz = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanılan cihaz:", cihaz)

VERI_KLASORU = Path("benim_verim")
SINIFLAR = [str(i) for i in range(10)] + ["X"]
ETIKET = {s: i for i, s in enumerate(SINIFLAR)}   # "0"->0 ... "9"->9, "X"->10
TEKRAR = 100   # kendi çizimlerimizi eğitimde kaç kez göstereceğiz


# 1) KENDİ VERİMİZ: her sınıfın %20'si teste ayrılır, model bunları eğitimde hiç görmez
egitim_ornek, test_ornek = [], []
for sinif in SINIFLAR:
    dosyalar = sorted((VERI_KLASORU / sinif).glob("*.png"))
    random.shuffle(dosyalar)
    k = max(1, len(dosyalar) // 5)
    test_ornek += [(d, ETIKET[sinif]) for d in dosyalar[:k]]
    egitim_ornek += [(d, ETIKET[sinif]) for d in dosyalar[k:]]
print(f"Kendi verim: {len(egitim_ornek)} eğitim, {len(test_ornek)} test çizimi")


class BenimVerim(Dataset):
    """benim_verim klasöründeki PNG çizimleri modele verilecek biçime getirir."""
    def __init__(self, ornekler, donusum):
        self.goruntuler = [Image.open(yol).convert("L") for yol, _ in ornekler]
        self.etiketler = [etiket for _, etiket in ornekler]
        self.donusum = donusum

    def __len__(self):
        return len(self.goruntuler)

    def __getitem__(self, i):
        return self.donusum(self.goruntuler[i]), self.etiketler[i]


# 2) VERİ ARTIRMA: her gösterişte çizimi biraz döndür, kaydır, büyüt/küçült
artirma = transforms.Compose([
    transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.85, 1.1)),
    transforms.ToTensor(),
])
duz = transforms.ToTensor()

mnist_egitim = datasets.MNIST("veri", train=True, download=True, transform=duz)
mnist_test = datasets.MNIST("veri", train=False, download=True, transform=duz)
benim_egitim = BenimVerim(egitim_ornek, artirma)
benim_test = BenimVerim(test_ornek, duz)

sentetik_x = BenimVerim([(d, 10) for d in sorted(Path("sentetik_X").glob("*.png"))], artirma)
print(f"Sentetik X görseli: {len(sentetik_x)}")
egitim_seti = ConcatDataset([mnist_egitim, sentetik_x] + [benim_egitim] * TEKRAR)
egitim_yukleyici = DataLoader(egitim_seti, batch_size=128, shuffle=True)
print(f"Toplam eğitim örneği (tur başına): {len(egitim_seti)}")


# 3) CNN MODELİ
class RakamCNN(nn.Module):
    def __init__(self, sinif_sayisi=11):
        super().__init__()
        # Özellik çıkarıcı: çizgi, köşe, kıvrım gibi şekilleri yakalar
        self.ozellik = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        # Karar verici: bulunan şekillerden hangi sınıf olduğuna karar verir
        self.karar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, sinif_sayisi),
        )

    def forward(self, x):
        return self.karar(self.ozellik(x))


model = RakamCNN().to(cihaz)
kayip_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def dogruluk(m, veri_seti):
    m.eval()
    dogru = 0
    with torch.no_grad():
        for x, y in DataLoader(veri_seti, batch_size=500):
            x, y = x.to(cihaz), y.to(cihaz)
            dogru += (m(x).argmax(1) == y).sum().item()
    return 100 * dogru / len(veri_seti)


# 4) EĞİTİM DÖNGÜSÜ — Ders 1'deki döngünün aynısı
for tur in range(5):
    model.train()
    for x, y in egitim_yukleyici:
        x, y = x.to(cihaz), y.to(cihaz)
        kayip = kayip_fn(model(x), y)
        optimizer.zero_grad()
        kayip.backward()
        optimizer.step()
    print(f"Tur {tur + 1}: kayıp={kayip.item():.3f}  "
          f"MNIST testi=%{dogruluk(model, mnist_test):.1f}  "
          f"Benim test çizimlerim=%{dogruluk(model, benim_test):.1f}")

torch.save(model.state_dict(), "rakam_cnn_modeli.pt")
print("Model kaydedildi: rakam_cnn_modeli.pt")


# 5) KARŞILAŞTIRMA: eski model ile yeni model, senin el yazınla
eski = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)).to(cihaz)
eski.load_state_dict(torch.load("rakam_modeli.pt", map_location=cihaz))

rakam_testi = BenimVerim([o for o in test_ornek if o[1] != 10], duz)   # sadece rakamlar
x_testi = BenimVerim([o for o in test_ornek if o[1] == 10], duz)       # sadece X çizimleri

print("\n=== KARŞILAŞTIRMA (modelin eğitimde hiç görmediği çizimlerin) ===")
print(f"Eski model (Ders 1, basit ağ, sadece MNIST): %{dogruluk(eski, rakam_testi):.1f}")
print(f"Yeni model (CNN + senin verin):              %{dogruluk(model, rakam_testi):.1f}")
print(f"'Rakam değil' çizimlerini tanıma:             %{dogruluk(model, x_testi):.1f}")

print("\nYeni modelin yanıldığı test çizimleri:")
model.eval()
hata_var = False
with torch.no_grad():
    for i in range(len(benim_test)):
        x, y = benim_test[i]
        tahmin = model(x.unsqueeze(0).to(cihaz)).argmax(1).item()
        if tahmin != y:
            hata_var = True
            print(f"  {SINIFLAR[y]} çizdim -> {SINIFLAR[tahmin]} dedi  ({test_ornek[i][0].name})")
if not hata_var:
    print("  Hiç hata yok!")
