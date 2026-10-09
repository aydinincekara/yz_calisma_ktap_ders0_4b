# Ders 2: Çocuklar rakam çizsin, kendi eğittiğimiz model tahmin etsin
# Not: rakam_modeli.pt (Ders 1'in çıktısı) bu dosyayla aynı klasörde olmalı.
# Sürüm 2: Belirsiz simge çubuğu yerine büyük "Tahtayı Temizle" düğmesi

import numpy as np
import streamlit as st
import torch
from torch import nn
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Rakam Tahmin Oyunu", page_icon="✏️")


@st.cache_resource
def model_yukle():
    # Ders 1'deki mimarinin AYNISI olmalı, yoksa ağırlıklar yüklenmez
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.load_state_dict(torch.load("rakam_modeli.pt", map_location="cpu"))
    model.eval()
    return model


def mnist_bicimine_cevir(rgba):
    """Çizimi, modelin eğitimde gördüğü biçime getirir:
    siyah zemin, beyaz rakam, 20x20 kutuya sığdırılmış, 28x28 ortasında."""
    gri = rgba[:, :, :3].mean(axis=2)
    satirlar = np.where(gri.max(axis=1) > 30)[0]
    sutunlar = np.where(gri.max(axis=0) > 30)[0]
    if len(satirlar) == 0:
        return None  # henüz çizim yok

    kirpik = gri[satirlar[0]:satirlar[-1] + 1, sutunlar[0]:sutunlar[-1] + 1]
    h, w = kirpik.shape
    olcek = 20 / max(h, w)
    yeni_boyut = (max(1, round(w * olcek)), max(1, round(h * olcek)))
    kucuk = Image.fromarray(kirpik.astype(np.uint8)).resize(yeni_boyut, Image.LANCZOS)

    tuval = Image.new("L", (28, 28), 0)
    tuval.paste(kucuk, ((28 - kucuk.width) // 2, (28 - kucuk.height) // 2))
    return np.array(tuval) / 255.0


model = model_yukle()

# Tahtayı temizlemek için sayaç: sayaç değişince tahta sıfırdan oluşturulur
if "tuval_no" not in st.session_state:
    st.session_state.tuval_no = 0

st.title("✏️ Rakam Tahmin Oyunu")
st.write("Siyah kutuya 0-9 arası bir rakam çiz, yapay zeka tahmin etsin! "
         "Yeniden çizmek için kutunun altındaki **Tahtayı Temizle** düğmesine bas.")

sol, sag = st.columns(2)

with sol:
    cizim = st_canvas(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        display_toolbar=False,                     # belirsiz simgeleri gizle
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
        st.info("Hadi bir rakam çiz!")
    else:
        x = torch.tensor(goruntu, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            olasiliklar = torch.softmax(model(x), dim=1)[0].numpy()

        tahmin = int(olasiliklar.argmax())
        st.markdown(f"## Tahminim: **{tahmin}**")
        st.write(f"Eminlik: %{olasiliklar[tahmin] * 100:.0f}")
        st.bar_chart({"olasılık": olasiliklar})
        st.caption("Modelin gördüğü 28x28 görüntü:")
        st.image(goruntu, width=112, clamp=True)
