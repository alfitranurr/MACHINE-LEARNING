# app.py
import streamlit as st
import os
import zipfile
import tempfile
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import pandas as pd

print(f"TensorFlow version: {tf.__version__}") 

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-header {
        font-size: 2rem;
        color: #2e86de;
        text-align: center;
        margin-bottom: 1rem;
    }
    .predict-box {
        background: #007bff;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #0056b3;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #bbdefb;
    }
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: 2px solid white;
    }
    .stSelectbox > div > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    zip_path = "MODEL.zip"
    if not os.path.exists(zip_path):
        st.error("MODEL.zip not found in the current directory!")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        model_dir = os.path.join(temp_dir, "MODEL")

        keg1_path = os.path.join(model_dir, "kegiatan1.h5")
        if os.path.exists(keg1_path):
            try:
                base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                base_model.trainable = False
                keg1_model = Sequential([
                    base_model,
                    GlobalAveragePooling2D(),
                    Dropout(0.2),
                    Dense(128, activation='relu'),
                    Dropout(0.2),
                    Dense(3, activation='softmax')
                ])
                keg1_model.load_weights(keg1_path)
                keg1_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                keg1_classes = ['paper', 'rock', 'scissors']
                print("Kegiatan1 model loaded successfully!")
            except Exception as e:
                print(f"Error loading kegiatan1: {e}")
                keg1_model = None
                keg1_classes = None
        else:
            print("kegiatan1.h5 not found!")
            keg1_model = None
            keg1_classes = None
        
        # Load kegiatan2 (Text)
        keg2_dir = os.path.join(model_dir, "kegiatan2")
        if os.path.exists(keg2_dir):
            try:
                tokenizer = AutoTokenizer.from_pretrained(keg2_dir)
                model = AutoModelForSequenceClassification.from_pretrained(keg2_dir)
                keg2_model = model
                keg2_tokenizer = tokenizer
                keg2_classes = ['negative', 'neutral', 'positive']
                print("Kegiatan2 model loaded successfully!")
            except Exception as e:
                print(f"Error loading kegiatan2: {e}")
                keg2_model = None
                keg2_tokenizer = None
                keg2_classes = None
        else:
            print("kegiatan2 directory not found!")
            keg2_model = None
            keg2_tokenizer = None
            keg2_classes = None

        keg3_dir = os.path.join(model_dir, "kegiatan3")
        if os.path.exists(keg3_dir):
            try:
                label_encoders = joblib.load(os.path.join(keg3_dir, "label_encoders.pkl"))
                feature_cols = joblib.load(os.path.join(keg3_dir, "feature_cols.pkl"))
                col_types = joblib.load(os.path.join(keg3_dir, "col_types.pkl"))
                
                tabnet_path = os.path.join(keg3_dir, "tabnet_model.zip")
                if not os.path.exists(tabnet_path):
                    tabnet_path = os.path.join(keg3_dir, "tabnet_model")
                
                print(f"Loading TabNet from: {tabnet_path}")
                
                keg3_model = TabNetClassifier()
                keg3_model.load_model(tabnet_path)
                
                keg3_feature_cols = feature_cols
                keg3_label_encoders = label_encoders
                keg3_col_types = col_types
                keg3_classes = ['<=50K', '>50K']
                print("Kegiatan3 model loaded successfully!")
            except Exception as e:
                print(f"Error loading kegiatan3: {e}")
                keg3_model = None
                keg3_feature_cols = None
                keg3_label_encoders = None
                keg3_col_types = None
                keg3_classes = None
        else:
            print("kegiatan3 directory not found!")
            keg3_model = None
            keg3_feature_cols = None
            keg3_label_encoders = None
            keg3_col_types = None
            keg3_classes = None
        
        return {
            'keg1_model': keg1_model, 'keg1_classes': keg1_classes,
            'keg2_model': keg2_model, 'keg2_tokenizer': keg2_tokenizer, 'keg2_classes': keg2_classes,
            'keg3_model': keg3_model, 'keg3_feature_cols': keg3_feature_cols,
            'keg3_label_encoders': keg3_label_encoders, 'keg3_col_types': keg3_col_types, 'keg3_classes': keg3_classes
        }

models = load_models()

st.markdown('<h1 class="main-header">🤖 Machine Learning Models Demo</h1>', unsafe_allow_html=True)
st.markdown("Pilih tab di bawah untuk mengakses model klasifikasi citra, teks, atau tabular!")

tab1, tab2, tab3 = st.tabs(["📸 Klasifikasi Citra", "📝 Analisis Sentimen", "📊 Prediksi Pendapatan"])

with tab1:
    st.markdown('<h2 class="tab-header">Rock-Paper-Scissors Image Classifier</h2>', unsafe_allow_html=True)
    st.info("📤 Upload gambar tangan (paper, rock, atau scissors) untuk prediksi instan!")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        uploaded_file = st.file_uploader("Pilih gambar...", type=['png', 'jpg', 'jpeg'])
    with col2:
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image = image.convert('RGB')  
                st.image(image, caption='Pratinjau Gambar', use_column_width=None)

                with st.spinner("🧠 Memproses gambar dengan AI..."):
                    img_array = np.array(image.resize((224, 224)))
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                if models['keg1_model']:
                    predictions = models['keg1_model'].predict(img_array)
                    predicted_class = models['keg1_classes'][np.argmax(predictions[0])]
                    confidence = np.max(predictions[0])
                    
                    st.markdown(f'<div class="predict-box"><h3>🎯 Hasil Prediksi: <strong>{predicted_class.upper()}</strong></h3><p><em>Akurasi Model:</em> <strong>{confidence:.2%}</strong></p></div>', unsafe_allow_html=True)
        
                    st.subheader("Distribusi Probabilitas:")
                    st.bar_chart(dict(zip(models['keg1_classes'], predictions[0])), height=250, use_container_width=True)
                else:
                    st.warning("⚠️ Model citra tidak tersedia.")
            except Exception as e:
                st.error(f"❌ Gagal memproses gambar: {e}")

with tab2:
    st.markdown('<h2 class="tab-header">Sentiment Analysis Text Classifier</h2>', unsafe_allow_html=True)
    st.info("💬 Masukkan ulasan atau teks untuk analisis sentimen (negative, neutral, positive).")

    text_input = st.text_area("Tulis teks Anda di sini...", height=200, placeholder="Contoh: 'Produk ini sangat bagus dan murah!'")
    predict_btn = st.button("🔍 Analisis Sentimen", type="primary")
    
    if predict_btn and text_input.strip():
        cleaned_text = text_input.lower().strip()
        
        if models['keg2_tokenizer'] and models['keg2_model']:
            with st.spinner("🧠 Menganalisis sentimen..."):
                inputs = models['keg2_tokenizer'](cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = models['keg2_model'](**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = models['keg2_classes'][torch.argmax(predictions, dim=1).item()]
                    confidence = torch.max(predictions).item()
  
            color = "🟢" if predicted_class == "positive" else "🟡" if predicted_class == "neutral" else "🔴"
            st.markdown(f'<div class="predict-box"><h3>{color} Sentimen: <strong>{predicted_class.upper()}</strong></h3><p><em>Keyakinan Model:</em> <strong>{confidence:.2%}</strong></p><small>Teks: "{text_input[:50]}..."</small></div>', unsafe_allow_html=True)
  
            st.subheader("Distribusi Sentimen:")
            probs = predictions[0].cpu().numpy()
            st.bar_chart(dict(zip(models['keg2_classes'], probs)), height=250, use_container_width=True)
        else:
            st.warning("⚠️ Model teks tidak tersedia.")
    elif predict_btn:
        st.warning("⚠️ Masukkan teks terlebih dahulu!")

with tab3:
    st.markdown('<h2 class="tab-header">Income Prediction Tabular Classifier</h2>', unsafe_allow_html=True)
    st.info("📋 Isi formulir data demografi untuk prediksi pendapatan (<=50K atau >50K). Gunakan nilai realistis untuk hasil akurat.")
    
    if models['keg3_feature_cols']:
        cat_cols = models['keg3_col_types']['cat_cols']
        num_cols = models['keg3_col_types']['num_cols']
    
        st.subheader("🔧 Form Input Data")
        input_data = {}
        for i, col in enumerate(models['keg3_feature_cols']):
            with st.container():
                cols = st.columns(2 if i % 2 == 0 else 1)  
                current_col = cols[0] if len(cols) > 1 else cols[0]
                if col in cat_cols:
                    options = list(models['keg3_label_encoders'][col].classes_)
                    input_data[col] = current_col.selectbox(f"**{col.replace('_', ' ').title()}**:", options, key=f"cat_{col}")
                elif col in num_cols:
                    input_data[col] = current_col.number_input(f"**{col.replace('_', ' ').title()}**:", value=0.0, step=1.0, format="%.0f", key=f"num_{col}")
                else:
                    input_data[col] = current_col.text_input(f"**{col.replace('_', ' ').title()}**:", value="", key=f"txt_{col}", placeholder="Masukkan nilai...")

        if st.button("🚀 Prediksi Pendapatan", type="primary"):
            if all(str(v) != '' and v is not None for v in input_data.values()):
                try:
                    df_input = pd.DataFrame([input_data])

                    for col in cat_cols:
                        df_input[col] = models['keg3_label_encoders'][col].transform([df_input[col].iloc[0]])
                    
                    X_input = df_input[models['keg3_feature_cols']].values    
                
                    with st.spinner("🧠 Memproses data tabular..."):
                        prediction = models['keg3_model'].predict(X_input)[0]
                        predicted_class = models['keg3_classes'][prediction]
                
                    icon = "💰" if predicted_class == ">50K" else "💼"
                    st.markdown(f'<div class="predict-box"><h3>{icon} Prediksi: <strong>{predicted_class}</strong></h3><p><em>Model TabNet menganalisis data Anda.</em></p></div>', unsafe_allow_html=True)
              
                    st.subheader("Probabilitas Pendapatan:")
                    probs = models['keg3_model'].predict_proba(X_input)[0]
                    st.bar_chart(dict(zip(models['keg3_classes'], probs)), height=250, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error predicting: {e}")
            else:
                st.warning("⚠️ Lengkapi semua field input!")
    else:
        st.warning("⚠️ Data fitur tidak dimuat. Periksa file model.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 0.9rem;'>Al Fitra Nur Ramadhani | Streamlit | © 2025</p>", unsafe_allow_html=True)