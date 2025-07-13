import streamlit as st
import numpy as np
from PIL import Image # Menggunakan PIL untuk memuat gambar di Streamlit
import tensorflow as tf # Import TensorFlow
from keras.models import load_model


loaded_model = load_model('model_fix.h5')

# --- Label mapping ---
class_labels = {
    0: "kualitas baik",
    1: "kualitas kurang"
}

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Kualitas Gambar",
    page_icon="🖼️",
    layout="centered"
)

st.title("Aplikasi Prediksi Kualitas Singkong")
st.write("Unggah gambar untuk memprediksi kualitasnya (baik atau kurang).")

# --- Pengunggah Gambar ---
uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_container_width=True)
    st.write("")
    st.write("Memprediksi...")

    # Memproses gambar
    # Mengubah ukuran gambar ke 224x224
    img_resized = image.resize((224, 224))
    # Mengubah gambar menjadi array NumPy
    img_array = np.array(img_resized)

    # Menambahkan dimensi batch karena model mengharapkan input dalam bentuk batch
    # Model Keras biasanya mengharapkan input dalam bentuk (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalisasi jika model Anda dilatih dengan data yang dinormalisasi (misal, dibagi 255.0)
    # Jika model Anda dilatih dengan input yang dinormalisasi, aktifkan baris di bawah ini:
    img_array = img_array / 255.0

    try:
        # Memprediksi kelas gambar
        predictions = loaded_model.predict(img_array)

        # Mengambil indeks kelas dengan probabilitas tertinggi
        predicted_class_index = np.argmax(predictions[0])

        # Mendapatkan nama kelas berdasarkan indeks yang diprediksi
        predicted_class_label = class_labels.get(predicted_class_index, "Tidak Diketahui")

        # Menampilkan hasil prediksi
        st.success(f'Gambar ini diprediksi memiliki: **{predicted_class_label}**')
        st.write(f"Probabilitas prediksi: {predictions[0][predicted_class_index]*100:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
        st.warning("Pastikan model Anda dimuat dengan benar dan input gambar sesuai dengan yang diharapkan model.")

st.markdown("---")
st.markdown("Dibuat dengan Streamlit dan TensorFlow.")
