# Ders 4B.1: Nesne algılama için sahne fotoğrafları topluyoruz
# Fotoğraflar yolo_verim/fotograflar/ klasörüne JPG olarak kaydedilir.

import time
from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Sahne Fotoğrafları", page_icon="🎯")

KLASOR = Path("yolo_verim") / "fotograflar"
KLASOR.mkdir(parents=True, exist_ok=True)
HEDEF_SAYI = 150

if "kamera_no" not in st.session_state:
    st.session_state.kamera_no = 0
if "son_foto" not in st.session_state:
    st.session_state.son_foto = None

st.title("🎯 Nesne Algılama: Sahne Fotoğrafları")
st.write("Masaya malzemeleri istediğin düzende yerleştir ve fotoğraf çek. "
         "**Her çekimden sonra düzeni değiştir.**")

sayi = len(list(KLASOR.glob("*.jpg")))
st.write(f"**{sayi} / {HEDEF_SAYI}** fotoğraf")
st.progress(min(sayi / HEDEF_SAYI, 1.0))

foto = st.camera_input("Sahneyi çek", key=f"sahne_{st.session_state.kamera_no}")

if foto is not None:
    yol = KLASOR / f"sahne_{int(time.time() * 1000)}.jpg"
    Image.open(foto).convert("RGB").save(yol, quality=92)
    st.session_state.son_foto = str(yol)
    st.session_state.kamera_no += 1   # kamerayı bir sonraki çekim için sıfırla
    st.rerun()

if st.session_state.son_foto and Path(st.session_state.son_foto).exists():
    st.success("Son kaydedilen fotoğraf:")
    st.image(st.session_state.son_foto, width=300)

st.caption(f"Fotoğrafların kaydedildiği yer: {KLASOR.resolve()}")
