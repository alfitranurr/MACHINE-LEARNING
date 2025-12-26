# app.py
import streamlit as st
import os
import zipfile
import tempfile
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D

print(f"TensorFlow version: {tf.__version__}") 
print(f"Current working directory: {os.getcwd()}")

# Set page config for wider layout and theme
st.set_page_config(
    page_title="Flower Image Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Overall app styling - Black background */
    .stApp {
        background: #000000; /* Black background */
        color: #ffffff; /* White text for contrast */
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff; /* White text */
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Subtitle */
    .subtitle {
        font-size: 2.5rem;
        font-weight: 400;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Sub headers */
    .tab-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #66bb6a; /* Lighter green */
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 0.5rem;
        border-bottom: 3px solid #2e7d32; /* Dark green border */
        background: rgba(76, 175, 80, 0.1); /* Subtle green tint */
        border-radius: 10px;
    }
    
    /* Predict box */
    .predict-box {
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%); /* Dark to green gradient */
        color: white;
        padding: 2rem;
        border-radius: 20px;
        border-left: 6px solid #66bb6a; /* Lighter green accent */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin: 1.5rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Image preview styling */
    .image-preview {
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        border: 3px solid #4caf50; /* Green border */
    }
    
    /* Bar chart container */
    .chart-container {
        background: #1a1a1a; /* Dark gray for contrast */
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        margin-top: 1rem;
        border: 1px solid #333333;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        color: white;
        border-radius: 25px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(76, 175, 80, 0.4);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #4caf50;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: #1a1a1a; /* Dark background */
        color: #ffffff;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%);
        border-radius: 15px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #4caf50;
    }
    .stTabs [data-baseweb="tab"] {
        color: #cccccc;
        font-weight: 500;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #4caf50;
        background: rgba(76, 175, 80, 0.1);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4caf50;
        background: rgba(76, 175, 80, 0.2);
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }
    
    /* Info boxes */
    .stInfo {
        background: #1a1a1a;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        color: #ffffff;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #aaaaaa;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 1rem;
        background: #1a1a1a;
        border-radius: 10px;
        border: 1px solid #333333;
    }
    
    /* Spinner customization */
    .stSpinner > div > div {
        border-top-color: #4caf50 !important;
        border-right-color: #4caf50 !important;
    }
    
    /* General text color override */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff;
    }
    
    /* Metric containers for better contrast */
    .stMetric {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # Search for MODEL.zip in common locations
    possible_paths = [
        "MODEL.zip",
        os.path.join("src", "MODEL.zip"),
        os.path.join("UAP", "src", "MODEL.zip"),
        os.path.join(os.getcwd(), "src", "MODEL.zip"),
        os.path.join(os.getcwd(), "MODEL.zip"),
        # TAMBAHAN: Path yang sesuai dengan struktur Anda (dari root repo ke UAP/streamlit-app/MODEL.zip)
        os.path.join("UAP", "streamlit-app", "MODEL.zip"),
        # Backup eksplisit dengan CWD
        os.path.join(os.getcwd(), "UAP", "streamlit-app", "MODEL.zip")
    ]
    
    # Debug: Print semua path yang dicoba (akan muncul di log Streamlit Cloud)
    print("Possible paths being checked:")
    for path in possible_paths:
        print(f"  - {path} (exists: {os.path.exists(path)})")
    
    zip_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if zip_path is None:
        st.error(f"MODEL.zip not found! Checked paths: {possible_paths}")
        st.stop()
    print(f"Using MODEL.zip at: {zip_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting to temp_dir: {temp_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Debug: List all files after extraction
        print("Files after extraction:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # Find the model directory dynamically (look for dir containing .keras files)
        model_dir = None
        for root, dirs, files in os.walk(temp_dir):
            if any(f.endswith('.keras') for f in files) and 'MODEL' in root.upper():
                model_dir = root
                break
        if model_dir is None:
            # Fallback to assumed path
            model_dir = os.path.join(temp_dir, "MODEL")
        
        print(f"Model dir set to: {model_dir}, exists: {os.path.exists(model_dir)}")

        def safe_load_model(model_path, model_name, build_func=None):
            if not os.path.exists(model_path):
                print(f"{model_name} path not found: {model_path}")
                return None
            try:
                # Coba load full model dulu (compile=False untuk hindari optimizer lama)
                model = keras.models.load_model(model_path, compile=False)
                model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
                print(f"{model_name} loaded successfully (full model)!")
                return model
            except Exception as e1:
                print(f"Full load failed for {model_name}: {e1}")
                if build_func:
                    try:
                        # Fallback: Build architecture baru, load weights dengan skip_mismatch=True
                        model = build_func()
                        model.load_weights(model_path, skip_mismatch=True)  # Skip layer dengan shape beda (e.g., classifier 256 vs 5)
                        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
                        print(f"{model_name} loaded with weights fallback (skip_mismatch=True)!")
                        return model
                    except Exception as e2:
                        print(f"Weights fallback failed for {model_name}: {e2}")
                        st.error(f"Error loading {model_name}: {e2}. Pastikan {model_path} valid dan classifier output 5 kelas.")
                        return None
                else:
                    print(f"No build_func for {model_name}, skipping.")
                    return None

        # Build functions untuk fallback (sesuai dengan training code)
        IMG_SIZE = (224, 224)
        NUM_CLASSES = 5

        def build_cnn_base():
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(NUM_CLASSES, activation='softmax')
            ])
            return model

        def build_mobilenetv2():
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
            base_model.trainable = False  
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dropout(0.3),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(256, activation='relu'),
                Dropout(0.2),
                Dense(NUM_CLASSES, activation='softmax')
            ])
            return model

        def build_resnet50():
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
            base_model.trainable = False  
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dropout(0.3),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(256, activation='relu'),
                Dropout(0.2),
                Dense(NUM_CLASSES, activation='softmax')
            ])
            return model

        # Load models dengan fallback
        cnn_path = os.path.join(model_dir, "cnn_base.keras")
        cnn_model = safe_load_model(cnn_path, "CNN Base", build_cnn_base)

        mobilenet_path = os.path.join(model_dir, "pretrained_mobilenetv2_no_finetune.keras")
        mobilenet_model = safe_load_model(mobilenet_path, "MobileNetV2", build_mobilenetv2)

        resnet_path = os.path.join(model_dir, "pretrained_resnet50_no_finetune.keras")
        resnet_model = safe_load_model(resnet_path, "ResNet50", build_resnet50)
        
        flower_classes = ['Tulip', 'Sunflower', 'Rose', 'Dandelion', 'Daisy']
        
        return {
            'cnn_model': cnn_model,
            'mobilenet_model': mobilenet_model,
            'resnet_model': resnet_model,
            'flower_classes': flower_classes
        }

models = load_models()

# Main title
st.markdown('<h1 class="main-header">🌸 Flower Image Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Klasifikasi Bunga Tulip, Sunflower, Rose, Dandelion, Daisy</p>', unsafe_allow_html=True)

# Display model status
st.sidebar.markdown("### Model Status")
if models['cnn_model']:
    st.sidebar.success("✅ CNN Base Loaded")
else:
    st.sidebar.error("❌ CNN Base Failed")
if models['mobilenet_model']:
    st.sidebar.success("✅ MobileNetV2 Loaded")
else:
    st.sidebar.error("❌ MobileNetV2 Failed")
if models['resnet_model']:
    st.sidebar.success("✅ ResNet50 Loaded")
else:
    st.sidebar.error("❌ ResNet50 Failed")

# Tabs for different models
tab_cnn, tab_mobilenet, tab_resnet = st.tabs([
    "🌼 Non-Pretrained (CNN)", 
    "🌻 Pretrained 1 (MobileNetV2)", 
    "🌹 Pretrained 2 (ResNet50)"
])

def predict_with_model(uploaded_file, model, classes, tab_name, preprocess_func):
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image = image.convert('RGB')  
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### Pratinjau Gambar")
                st.image(image, caption=f'Gambar untuk {tab_name}', clamp=True)

            with col2:
                with st.spinner(f"🧠 AI sedang menganalisis gambar dengan {tab_name}..."):
                    img_array = np.array(image.resize((224, 224)), dtype=np.float32)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_func(img_array)
                
                if model:
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class = classes[np.argmax(predictions[0])]
                    confidence = np.max(predictions[0])
                    
                    st.markdown(f'''
                    <div class="predict-box">
                        <h3>🎯 Hasil Prediksi: <strong>{predicted_class}</strong></h3>
                        <p><em>Confidence:</em> <strong>{confidence:.2%}</strong></p>
                        <small>Diklasifikasikan oleh {tab_name}</small>
                    </div>
                    ''', unsafe_allow_html=True)
            
                    st.subheader("📊 Distribusi Probabilitas")
                    st.bar_chart(
                        dict(zip(classes, predictions[0])), 
                        height=300, 
                        use_container_width=True
                    )
                else:
                    st.warning(f"⚠️ Model {tab_name} tidak tersedia. Periksa file MODEL.zip.")
        except Exception as e:
            st.error(f"❌ Gagal memproses gambar: {str(e)}")
    else:
        st.info("📤 Upload gambar bunga (Tulip, Sunflower, Rose, Dandelion, atau Daisy) untuk prediksi instan!")

with tab_cnn:
    st.markdown('<h2 class="tab-header">Non-Pretrained (CNN)</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        uploaded_file = st.file_uploader("Pilih gambar...", type=['png', 'jpg', 'jpeg'], key="cnn_uploader")
    with col2:
        def cnn_preprocess(x):
            return x / 255.0
        predict_with_model(uploaded_file, models['cnn_model'], models['flower_classes'], "CNN Base", cnn_preprocess)

with tab_mobilenet:
    st.markdown('<h2 class="tab-header">Pretrained 1 (MobileNetV2)</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        uploaded_file = st.file_uploader("Pilih gambar...", type=['png', 'jpg', 'jpeg'], key="mobilenet_uploader")
    with col2:
        predict_with_model(uploaded_file, models['mobilenet_model'], models['flower_classes'], "MobileNetV2", mobilenet_preprocess)

with tab_resnet:
    st.markdown('<h2 class="tab-header">Pretrained 2 (ResNet50)</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        uploaded_file = st.file_uploader("Pilih gambar...", type=['png', 'jpg', 'jpeg'], key="resnet_uploader")
    with col2:
        predict_with_model(uploaded_file, models['resnet_model'], models['flower_classes'], "ResNet50", resnet_preprocess)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🌟 Dibuat menggunakan Streamlit | Al Fitra Nur Ramadhani | © 2025</p>
</div>
""", unsafe_allow_html=True)