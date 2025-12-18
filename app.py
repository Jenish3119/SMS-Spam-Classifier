import streamlit as st
import joblib
import re
import os
from PIL import Image

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Spam Message Detection",
    page_icon="📩",
    layout="centered"
)

# ================= PATH HANDLING =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
HAM_WC_PATH = os.path.join(BASE_DIR, "ham_wc.png")
SPAM_WC_PATH = os.path.join(BASE_DIR, "spam_wc.png")

# ================= LOAD MODEL & VECTORIZER =================
@st.cache_resource
def load_objects():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_objects()

# ================= PREPROCESS FUNCTION =================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    return " ".join(text)

# ================= IMAGE SAFE DISPLAY =================
def show_image(path, caption):
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_column_width=True)
    else:
        st.info(f"{caption} not available")

# ================= SIDEBAR =================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Try the Model", "Dataset Insights", "About"]
)

# ================= HOME =================
if page == "Home":
    st.title("📩 Spam Message Detection System")

    st.markdown("""
    This project demonstrates an **end-to-end NLP-based Spam Detection System**
    deployed as a **production-ready web application**.

    ### 🔍 Key Features
    - Text preprocessing & cleaning  
    - TF-IDF feature extraction  
    - Supervised machine learning classification  
    - Real-time message prediction  
    - Automated & cloud-ready deployment  
    """)

# ================= TRY THE MODEL =================
elif page == "Try the Model":
    st.title("🧪 Try the Spam Classifier")

    user_text = st.text_area("✉️ Enter your message")

    if st.button("Predict"):
        if user_text.strip() == "":
            st.warning("Please enter a message")
        else:
            clean_text = preprocess(user_text)
            vector = vectorizer.transform([clean_text])
            prediction = model.predict(vector)
            confidence = model.predict_proba(vector)

            if prediction[0] == 1:
                st.error(
                    f"🚨 SPAM Message | Confidence: {confidence[0][1]*100:.2f}%"
                )
            else:
                st.success(
                    f"✅ HAM Message | Confidence: {confidence[0][0]*100:.2f}%"
                )

# ================= DATASET INSIGHTS =================
elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")

    st.markdown("### 🔹 WordCloud Analysis")

    col1, col2 = st.columns(2)

    with col1:
        show_image(HAM_WC_PATH, "Ham Messages WordCloud")

    with col2:
        show_image(SPAM_WC_PATH, "Spam Messages WordCloud")

    st.markdown("""
    **Observations**
    - Spam messages commonly include promotional terms like *free, win, offer*
    - Ham messages are more conversational and personal
    """)

# ================= ABOUT =================
else:
    st.title("ℹ️ About This Project")

    st.markdown("""
    **Spam Message Detection System**

    **Technologies Used**
    - Python
    - Scikit-learn
    - Natural Language Processing (NLP)
    - Streamlit

    **Pipeline**
    - Text preprocessing  
    - TF-IDF vectorization  
    - Supervised classification  
    - Model serialization  
    - Automated deployment  

    **Key Focus**
    - Reusability  
    - Robust error handling  
    - Cloud compatibility  
    """)
