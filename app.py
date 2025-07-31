import streamlit as st
import base64
import numpy as np
import cv2
from deepface import DeepFace
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Konfigurasi
st.set_page_config(page_title="Stres Analyzer", layout="wide")

# Fungsi bantu
def decode_image(uploaded_file):
    bytes_data = uploaded_file.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def analyze_emotion(img):
    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=True)
        emotion = result[0]['dominant_emotion'].lower()
        emotion_score_map = {
            'angry': 90, 'fear': 85, 'disgust': 80,
            'sad': 75, 'neutral': 50, 'surprise': 40, 'happy': 20
        }
        return emotion_score_map.get(emotion, 50), emotion
    except:
        return 50, "Tidak terdeteksi"

def get_ai_solution(stress_score, label, jawaban):
    prompt = f"""
    Anda adalah seorang konselor psikologi AI yang empatik, bijak, dan suportif dalam bahasa Indonesia.
    Berikut ringkasan:
    - Tingkat stres: {label} ({stress_score}%)
    - Pikiran dominan: {jawaban['q1']}
    - Durasi: {jawaban['q2']}
    - Emosi: {jawaban['q3']}
    - Tidur: {jawaban['q4']}
    - Cara mengatasi: {jawaban['q5']}
    - Dukungan sosial: {jawaban['q6']}
    - Kebutuhan: {jawaban['q7']}
    
    1. Validasi perasaan mereka secara empatik.
    2. Berikan 2-3 saran praktis.
    3. Akhiri dengan penyemangat.
    Format poin dan jangan berikan disclaimer profesional.
    """
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text

# ================== UI ===================

st.title("🧠 Deteksi Tingkat Stres Berdasarkan Wajah & Kuesioner")

uploaded_img = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
if uploaded_img:
    img = decode_image(uploaded_img)
    st.image(img, caption="Wajah Anda", channels="BGR")

    with st.spinner("Mendeteksi emosi..."):
        emotion_score, dominant_emotion = analyze_emotion(img)

    st.success(f"Emosi dominan: **{dominant_emotion}** (skor: {emotion_score})")

    st.header("📝 Jawab Pertanyaan Berikut")
    options = {
        'q1': st.selectbox("Apa yang paling membebani pikiranmu akhir-akhir ini?", 
                           ['Pekerjaan / tugas', 'Hubungan sosial / keluarga', 'Finansial atau Kesehatan']),
        'q2': st.selectbox("Sejak kapan kamu merasa seperti ini?",
                           ['Beberapa hari', 'Lebih dari seminggu', 'Beberapa bulan', 'Sangat lama, bahkan lupa kapan merasa baik']),
        'q3': st.selectbox("Apa yang paling sering kamu rasakan?",
                           ['Cemas / khawatir berlebihan', 'Marah / mudah tersinggung', 'Lelah atau Mati rasa']),
        'q4': st.selectbox("Bagaimana kualitas tidurmu akhir-akhir ini?",
                           ['Nyenyak & cukup', 'Sering bangun / gelisah', 'Sulit tidur', 'Terlalu banyak tidur']),
        'q5': st.selectbox("Bagaimana cara kamu mengatasi tekanan tersebut?",
                           ['Curhat ke teman / keluarga', 'Menyibukkan diri', 'Menarik diri dari sekitar', 'Bingung harus bagaimana']),
        'q6': st.selectbox("Apakah kamu merasa punya dukungan sosial?", 
                           ['Ya, lebih dari satu', 'Ya, tapi hanya satu-dua', 'Tidak yakin', 'Tidak ada']),
        'q7': st.selectbox("Apa yang paling kamu butuhkan saat ini?",
                           ['Waktu istirahat atau Tempat curhat', 'Arahan atau solusi praktis', 'Tidak tahu, tapi ingin merasa lebih baik'])
    }

    # Mapping ke skor
    skor_map = {
        'q1': {'Pekerjaan / tugas': 10, 'Hubungan sosial / keluarga': 15, 'Finansial atau Kesehatan': 20},
        'q2': {'Beberapa hari': 5, 'Lebih dari seminggu': 10, 'Beberapa bulan': 15, 'Sangat lama, bahkan lupa kapan merasa baik': 20},
        'q3': {'Cemas / khawatir berlebihan': 15, 'Marah / mudah tersinggung': 10, 'Lelah atau Mati rasa': 20},
        'q4': {'Nyenyak & cukup': 0, 'Sering bangun / gelisah': 10, 'Sulit tidur': 15, 'Terlalu banyak tidur': 20},
        'q5': {'Curhat ke teman / keluarga': 0, 'Menyibukkan diri': 10, 'Menarik diri dari sekitar': 20, 'Bingung harus bagaimana': 15},
        'q6': {'Ya, lebih dari satu': 0, 'Ya, tapi hanya satu-dua': 5, 'Tidak yakin': 15, 'Tidak ada': 20},
        'q7': {'Waktu istirahat atau Tempat curhat': 5, 'Arahan atau solusi praktis': 10, 'Tidak tahu, tapi ingin merasa lebih baik': 15}
    }

    q_score = sum(skor_map[k].get(v, 0) for k, v in options.items())
    final_score = int((0.6 * (q_score / 135 * 100)) + (0.4 * emotion_score))

    if final_score >= 70:
        label = "Stres Berat"
    elif final_score >= 45:
        label = "Stres Sedang"
    else:
        label = "Stres Ringan"

    if st.button("💡 Lihat Hasil dan Saran AI"):
        st.subheader(f"Tingkat Stres Anda: {label} ({final_score}%)")
        with st.spinner("Menghasilkan saran..."):
            solution = get_ai_solution(final_score, label, options)
        st.markdown("### 💬 Saran dari AI:")
        st.markdown(solution)
