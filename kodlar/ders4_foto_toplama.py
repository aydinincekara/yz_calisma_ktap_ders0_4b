# Ders 4.1: Webcam ile atölye malzemesi fotoğrafları topluyoruz
# Fotoğraflar atolye_verim/<sınıf>/ klasörlerine JPG olarak kaydedilir.

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fotoğraf Toplama", page_icon="📸")

VERI_KLASORU = Path("atolye_verim")
SINIFLAR = {
    "arduino_uno": "Arduino Uno",
    "breadboard": "Breadboard",
    "ultrasonik_sensor": "Ultrasonik Sensör (HC-SR04)",
    "servo_motor": "Servo Motor",
    "hicbiri": "Hiçbiri (malzeme yok)",
}
HEDEF_SAYI = 50


def foto_sayilari():
    return {s: len(list((VERI_KLASORU / s).glob("*.jpg"))) for s in SINIFLAR}


# Durum bilgileri: kamera sayacı (her çekimden sonra kamera sıfırlanır) ve son fotoğraf
if "kamera_no" not in st.session_state:
    st.session_state.kamera_no = 0
if "son_foto" not in st.session_state:
    st.session_state.son_foto = None

st.title("📸 Atölye Malzemesi Fotoğraf Toplama")

sinif = st.selectbox("Hangi malzemeyi çekiyorsun?", list(SINIFLAR),
                     format_func=lambda s: SINIFLAR[s])
sayilar = foto_sayilari()
st.write(f"**{SINIFLAR[sinif]}**: {sayilar[sinif]} / {HEDEF_SAYI} fotoğraf")
st.progress(min(sayilar[sinif] / HEDEF_SAYI, 1.0))

foto = st.camera_input("Malzemeyi kameraya göster ve fotoğraf çek",
                       key=f"kamera_{st.session_state.kamera_no}")

if foto is not None:
    klasor = VERI_KLASORU / sinif
    klasor.mkdir(parents=True, exist_ok=True)
    yol = klasor / f"{int(time.time() * 1000)}.jpg"
    Image.open(foto).convert("RGB").save(yol, quality=90)
    st.session_state.son_foto = (str(yol), SINIFLAR[sinif])
    st.session_state.kamera_no += 1   # kamerayı bir sonraki çekim için sıfırla
    st.rerun()

if st.session_state.son_foto and Path(st.session_state.son_foto[0]).exists():
    yol, ad = st.session_state.son_foto
    st.success(f"Son kaydedilen fotoğraf: {ad}")
    st.image(yol, width=200)

st.divider()
st.subheader("Toplanan fotoğraflar")
st.bar_chart(pd.DataFrame({"fotoğraf sayısı": list(sayilar.values())},
                          index=[SINIFLAR[s] for s in SINIFLAR]))
st.write(f"Toplam **{sum(sayilar.values())}** fotoğraf. "
         f"Hedef: her malzemeden **{HEDEF_SAYI}** fotoğraf.")
