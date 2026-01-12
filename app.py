import streamlit as st
import joblib
import re
import string
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ------------------- NLTK SETUP -------------------
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

download_nltk()

# ------------------- LOAD MODEL & ARTIFAK -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(os.path.join(BASE_DIR, "model/tfidf_vectorizer.pkl"))
    model = joblib.load(os.path.join(BASE_DIR, "model/logreg_model.pkl"))
    slang = joblib.load(os.path.join(BASE_DIR, "model/slangwords.pkl"))
    return tfidf, model, slang

tfidf, model, slangwords = load_artifacts()

# ------------------- PREPROCESSING -------------------
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+','', text)
    text = re.sub(r'#[A-Za-z0-9]+','', text)
    text = re.sub(r'RT[\s]','', text)
    text = re.sub(r"http\S+",'', text)
    text = re.sub(r'[0-9]+','', text)
    text = re.sub(r'[^\w\s]','', text)
    text = text.replace('\n',' ')
    text = text.translate(str.maketrans('','',string.punctuation))
    return text.strip()

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    words = text.split()
    return ' '.join([slangwords.get(w, w) for w in words])

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(tokens):
    stop_words = set(stopwords.words("indonesian") + stopwords.words("english"))
    extra = {'iya','yaa','gak','nya','na','sih','ku','di','ga','ya','gaa','loh','kah'}
    stop_words.update(extra)
    return [w for w in tokens if w not in stop_words]

def toSentence(tokens):
    return ' '.join(tokens)

def preprocess(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    text = fix_slangwords(text)
    tokens = tokenizingText(text)
    tokens = filteringText(tokens)
    return toSentence(tokens)

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Analisis Sentimen by.u", layout="centered")

st.title("Analisis Sentimen Ulasan by.u")
st.caption("Masukkan satu ulasan, lalu tekan **Predict**")

user_input = st.text_area(
    "Ulasan",
    height=150,
    placeholder="Contoh: aplikasinya bagus banget, cepet, gampang topup"
)

if st.button("Predict", type="primary"):
    if not user_input.strip():
        st.warning("Tulis ulasan dulu ya!")
    else:
        with st.spinner("Sedang memproses..."):
            clean_text = preprocess(user_input)
            vec = tfidf.transform([clean_text])
            pred = model.predict(vec)[0]
            score = model.predict_proba(vec)[0].max()

        st.success("**SELESAI!**")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Prediksi", pred.upper())
        with col2:
            st.progress(score)
            st.caption(f"Confidence: {score:.1%}")

        with st.expander("Lihat hasil preprocessing"):
            st.write(clean_text)
