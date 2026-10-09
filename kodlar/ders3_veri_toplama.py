# Ders 3.1: Kendi veri setimizi topluyoruz
# Çizimler benim_verim/<sınıf>/ klasörlerine 28x28 PNG olarak kaydedilir.
# Sınıflar: 0-9 rakamları + "X" (rakam olmayan çizimler)

import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Veri Toplama", page_icon="📦")

VERI_KLASORU = Path("benim_verim")
SINIFLAR = [str(i) for i in range(10)] + ["X"]


def mnist_bicimine_cevir(rgba):
    """Ders 2'deki fonksiyonun aynısı: siyah zemin, beyaz çizim, 28x28, ortalanmış."""
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


def ornek_sayilari():
    return {s: len(list((VERI_KLASORU / s).glob("*.png"))) for s in SINIFLAR}


# Durum bilgileri: tahta sayacı ve sıradaki görev
if "tuval_no" not in st.session_state:
    st.session_state.tuval_no = 0
sayilar = ornek_sayilari()
if "hedef" not in st.session_state:
    st.session_state.hedef = min(sayilar, key=sayilar.get)  # en az örneği olan sınıf
secim = st.sidebar.selectbox("Görev seçimi", ["Otomatik"] + SINIFLAR)
hedef = st.session_state.hedef if secim == "Otomatik" else secim

st.title("📦 Rakam Veri Toplama")
if hedef == "X":
    st.markdown("## Görev: Rakam **OLMAYAN** bir şey çiz")
    st.write("Kalp, yıldız, harf, ev, karalama... 0-9 dışında ne istersen.")
else:
    st.markdown(f"## Görev: **{hedef}** rakamını çiz")
    st.write("Kendi yazdığın gibi çiz. Türk usulü 7 ve 1 de olur!")

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
        key=f"toplama_{st.session_state.tuval_no}",
    )

with sag:
    goruntu = None
    if cizim.image_data is not None:
        goruntu = mnist_bicimine_cevir(cizim.image_data)

    if goruntu is None:
        st.info("Soldaki siyah kutuya çiz.")
    else:
        st.caption("Kaydedilecek 28x28 görüntü:")
        st.image(goruntu, width=112, clamp=True)

    kaydet = st.button("💾 Kaydet ve sıradaki görev", type="primary",
                       use_container_width=True, disabled=goruntu is None)
    temizle = st.button("🗑️ Tahtayı Temizle", use_container_width=True)

if kaydet:
    klasor = VERI_KLASORU / hedef
    klasor.mkdir(parents=True, exist_ok=True)
    dosya_adi = klasor / f"{int(time.time() * 1000)}.png"
    Image.fromarray((goruntu * 255).astype(np.uint8)).save(dosya_adi)
    sayilar[hedef] += 1
    st.session_state.hedef = min(sayilar, key=sayilar.get)
    st.session_state.tuval_no += 1
    st.rerun()

if temizle:
    st.session_state.tuval_no += 1
    st.rerun()

st.divider()
st.subheader("Toplanan veri")
st.bar_chart(pd.DataFrame({"çizim sayısı": list(sayilar.values())}, index=SINIFLAR))
st.write(f"Toplam **{sum(sayilar.values())}** çizim. Hedef: her sınıftan en az **20** çizim.")
