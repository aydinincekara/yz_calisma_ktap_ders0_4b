# Ders 3.3: Rakam Tahmin Oyunu v2
# Yenilikler: CNN modeli, "rakam değil" cevabı, ayarlanabilir "emin değilim" eşiği
# Not: rakam_cnn_modeli.pt (Ders 3.2'nin çıktısı) bu dosyayla aynı klasörde olmalı.

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Rakam Tahmin Oyunu v2", page_icon="✏️")

SINIFLAR = [str(i) for i in range(10)] + ["X"]


class RakamCNN(nn.Module):
    """ders3_cnn_egitimi.py içindeki modelin AYNISI olmalı, yoksa ağırlıklar yüklenmez."""
    def __init__(self, sinif_sayisi=11):
        super().__init__()
        self.ozellik = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.karar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, sinif_sayisi),
        )

    def forward(self, x):
        return self.karar(self.ozellik(x))


@st.cache_resource
def model_yukle():
    model = RakamCNN()
    model.load_state_dict(torch.load("rakam_cnn_modeli.pt", map_location="cpu"))
    model.eval()
    return model


def mnist_bicimine_cevir(rgba):
    """Çizimi, modelin eğitimde gördüğü biçime getirir:
    siyah zemin, beyaz rakam, 20x20 kutuya sığdırılmış, 28x28 ortasında."""
    gri = rgba[:, :, :3].mean(axis=2)
    satirlar = np.where(gri.max(axis=1) > 30)[0]
    sutunlar = np.where(gri.max(axis=0) > 30)[0]
    if len(satirlar) == 0:
        return None

    kirpik = gri[satirlar[0]:satirlar[-1] + 1, sutunlar[0]:sutunlar[-1] + 1]
    h, w = kirpik.shape
    olcek = 20 / max(h, w)
    yeni_boyut = (max(1, round(w * olcek)), max(1, round(h * olcek)))
    kucuk = Image.fromarray(kirpik.astype(np.uint8)).resize(yeni_boyut, Image.LANCZOS)

    tuval = Image.new("L", (28, 28), 0)
    tuval.paste(kucuk, ((28 - kucuk.width) // 2, (28 - kucuk.height) // 2))
    return np.array(tuval) / 255.0


model = model_yukle()

if "tuval_no" not in st.session_state:
    st.session_state.tuval_no = 0

# Kenar çubuğu: eminlik eşiği
st.sidebar.header("⚙️ Ayarlar")
esik = st.sidebar.slider("Eminlik eşiği (%)", min_value=30, max_value=95, value=60, step=5)
st.sidebar.write("Model bu değerden daha az eminse tahmin vermez, **'Emin değilim'** der.")

st.title("✏️ Rakam Tahmin Oyunu v2")
st.write("Siyah kutuya bir rakam çiz, yapay zeka tahmin etsin! "
         "Rakam olmayan bir şey çizersen bunu da anlamaya çalışacak.")

sol, sag = st.columns(2)

with sol:
    cizim = st_canvas(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        display_toolbar=False,
        key=f"tuval_{st.session_state.tuval_no}",
    )
    if st.button("🗑️ Tahtayı Temizle", type="primary", use_container_width=True):
        st.session_state.tuval_no += 1
        st.rerun()

with sag:
    goruntu = None
    if cizim.image_data is not None:
        goruntu = mnist_bicimine_cevir(cizim.image_data)

    if goruntu is None:
        st.info("Hadi bir şey çiz!")
    else:
        # CNN görüntüyü (adet, kanal, yükseklik, genişlik) biçiminde ister: (1, 1, 28, 28)
        x = torch.tensor(goruntu, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            olasiliklar = torch.softmax(model(x), dim=1)[0].numpy()

        sirali = olasiliklar.argsort()[::-1]
        birinci, ikinci = sirali[0], sirali[1]
        eminlik = olasiliklar[birinci] * 100

        if eminlik < esik:
            st.markdown("## 🤔 Emin değilim")
            st.write(f"Aklımdakiler: **{SINIFLAR[birinci]}** (%{eminlik:.0f}) "
                     f"veya **{SINIFLAR[ikinci]}** (%{olasiliklar[ikinci] * 100:.0f})")
        elif SINIFLAR[birinci] == "X":
            st.markdown("## 🙅 Bu bir rakam değil!")
            st.write(f"Eminlik: %{eminlik:.0f}")
        else:
            st.markdown(f"## Tahminim: **{SINIFLAR[birinci]}**")
            st.write(f"Eminlik: %{eminlik:.0f}")

        st.bar_chart(pd.DataFrame({"olasılık": olasiliklar}, index=SINIFLAR))
        st.caption("Grafikteki X = rakam değil. Modelin gördüğü 28x28 görüntü:")
        st.image(goruntu, width=112, clamp=True)
