# Ders 1: Sıfırdan ilk yapay zeka modelim — el yazısı rakam tanıma (0-9)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

cihaz = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanılan cihaz:", cihaz)

# 1) VERİ: 60.000 eğitim + 10.000 test görseli (28x28 piksel, siyah-beyaz)
donusum = transforms.ToTensor()
egitim = datasets.MNIST("veri", train=True, download=True, transform=donusum)
test = datasets.MNIST("veri", train=False, download=True, transform=donusum)
egitim_yukleyici = DataLoader(egitim, batch_size=64, shuffle=True)
test_yukleyici = DataLoader(test, batch_size=1000)

# 2) MODEL: 784 piksel -> 128 nöron -> 10 çıktı (her rakam için bir puan)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
).to(cihaz)

# 3) ÖĞRENME AYARLARI
kayip_fn = nn.CrossEntropyLoss()                            # "Ne kadar yanıldım?"
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # "Ağırlıkları nasıl düzelteyim?"


def dogruluk_olc():
    model.eval()
    dogru = 0
    with torch.no_grad():
        for x, y in test_yukleyici:
            x, y = x.to(cihaz), y.to(cihaz)
            dogru += (model(x).argmax(1) == y).sum().item()
    return 100 * dogru / len(test)


print(f"Eğitimden ÖNCE test doğruluğu: %{dogruluk_olc():.1f}")

# 4) EĞİTİM DÖNGÜSÜ — yapay zekanın kalbi
for tur in range(3):
    model.train()
    for x, y in egitim_yukleyici:
        x, y = x.to(cihaz), y.to(cihaz)
        tahmin = model(x)              # a) Tahmin et
        kayip = kayip_fn(tahmin, y)    # b) Hatayı ölç
        optimizer.zero_grad()
        kayip.backward()               # c) Hatanın kaynağını bul (geri yayılım)
        optimizer.step()               # d) Ağırlıkları biraz düzelt
    print(f"Tur {tur + 1}: kayıp={kayip.item():.3f}  test doğruluğu=%{dogruluk_olc():.1f}")

# 5) KAYDET
torch.save(model.state_dict(), "rakam_modeli.pt")
print("Model kaydedildi: rakam_modeli.pt")
