import streamlit as st
import re
import nltk
import numpy as np
from keras.models import load_model
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Pastikan stopwords sudah terunduh satu kali
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fungsi untuk membersihkan teks
def preprocess_text(input_text):
    input_text = re.sub(r'[^a-zA-Z\s]', '', input_text)  # Menghapus tanda baca dan angka
    input_text = input_text.lower()  # Mengubah teks menjadi huruf kecil
    input_text = ' '.join([word for word in input_text.split() if word not in stop_words])  # Menghapus stop words
    return input_text

# Muat model neural network Keras dan vektorizer
model = load_model('bullying_model2.h5')  # Pastikan model disimpan sebagai .h5
vectorizer = joblib.load('vectorizer.pkl')  # Gunakan vektorizer yang sesuai saat pelatihan

# Aplikasi Streamlit
st.title("Cyberbullying Detection App")
st.write("Masukkan tweet untuk memprediksi apakah itu merupakan tweet bullying atau bukan.")

input_text = st.text_area("Input Tweet")
if st.button("Prediksi"):
    cleaned_text = preprocess_text(input_text)
    transformed_text = vectorizer.transform([cleaned_text]).toarray()  # Pastikan transformasi sesuai input neural network
    prediction = model.predict(transformed_text)
    result = 'Cyberbullying' if prediction[0][0] > 0.5 else 'Not Cyberbullying'
    st.write(f"Hasil Prediksi: {result}")
    st.write(f"Probabilitas: {prediction[0][0]:.2f}")
