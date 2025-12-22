# Flower Image Classification

## Deskripsi Proyek

Proyek ini merupakan implementasi sistem klasifikasi gambar bunga menggunakan tiga jenis model deep learning: 
- **CNN Base (Non-Pretrained)**: Model Convolutional Neural Network yang dibangun dari awal tanpa bobot pre-trained.
- **Pre-Trained MobileNetV2 (No Fine-Tuning)**: Model transfer learning berbasis MobileNetV2 dengan bobot ImageNet, di mana lapisan base dibekukan (tidak di-fine-tune).
- **Pre-Trained ResNet50 (No Fine-Tuning)**: Model transfer learning berbasis ResNet50 dengan bobot ImageNet, di mana lapisan base dibekukan (tidak di-fine-tune).

Dataset yang digunakan adalah kumpulan gambar bunga dengan 5 kelas: **tulip**, **sunflower**, **rose**, **dandelion**, dan **daisy**. Proyek ini mencakup preprocessing data, augmentasi, pelatihan model, evaluasi performa, dan deploy sebagai aplikasi web menggunakan Streamlit untuk memungkinkan pengguna mengunggah gambar bunga dan mendapatkan prediksi kelas secara real-time.

Proyek ini dikembangkan menggunakan Python, TensorFlow/Keras, dan dijalankan di Google Colab untuk pelatihan, dengan model disimpan ke Google Drive. Aplikasi web Streamlit memungkinkan demo lokal atau deployment.

Struktur folder proyek (berdasarkan gambar yang diberikan):
```
UAP/
├── __pycache__/
├── venv/
├── streamlit-app/          # Folder untuk aplikasi Streamlit (app.py, requirements.txt)
├── MODEL.zip               # Arsip model yang dilatih
├── requirements.old.txt    # Requirements lama
├── requirements.txt        # Requirements untuk Streamlit app
├── .gitignore
├── pdm-python
├── pyproject.toml
├── README.md               # File ini
└── UAP MODUL PEMBELAJARAN MESIN.pdf  # Dokumen modul
```

## Penjelasan Dataset dan Preprocessing

### Dataset
- **Sumber**: Dataset gambar bunga dari Google Drive (`/content/drive/MyDrive/STUDI UMM/SEMESTER 7/PRAKTIKUM/UAP/DATASET/flower/`).
- **Kelas**: 5 kelas bunga – tulip, sunflower, rose, dandelion, daisy.
- **Distribusi Data Asli**:
  - tulip: 607 gambar
  - sunflower: 495 gambar
  - rose: 497 gambar
  - dandelion: 646 gambar
  - daisy: 501 gambar
- **Total**: 2.746 gambar.
- **Ukuran Gambar Sample**: Bervariasi, misalnya (320x240), (500x333), (240x216), dll. Semua gambar di-resize ke (224x224) untuk input model.
- **Split Dataset**: 
  - Train: 70% (~1.919 gambar asli, setelah augmentasi: 5.500 gambar).
  - Validation: 15% (410 gambar).
  - Test: 15% (417 gambar).

### Preprocessing
1. **Pembagian Dataset**: Data dibagi secara acak menggunakan `train_test_split`-like logic dengan rasio 70-15-15, dan disalin ke struktur folder `train/`, `val/`, `test/` per kelas.
2. **Augmentasi Data (Hanya untuk Train)**: Meningkatkan data train hingga ~1.100 gambar per kelas menggunakan `ImageDataGenerator`:
   - Rotasi: 15°
   - Shift: 0.15 (width/height)
   - Flip horizontal: True
   - Zoom: 0.15
   - Brightness: [0.85, 1.15]
   - Rescale: 1./255 (untuk CNN base).
   - Total train setelah augmentasi: 5.500 gambar (1.100 per kelas).
3. **Data Generator**:
   - Batch size: 16.
   - Target size: (224, 224).
   - Untuk pre-trained models: `preprocess_input` dari ResNet50/MobileNetV2.
   - Augmentasi tambahan pada train generator (sama seperti di atas).
4. **Visualisasi**: Distribusi kelas divisualisasikan dengan bar chart, dan sample gambar ditampilkan untuk verifikasi.

## Penjelasan Ketiga Model yang Digunakan

Semua model dilatih selama 20 epoch dengan optimizer Adam, loss `categorical_crossentropy`, dan callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint). Model terbaik disimpan sebagai `.keras` di `/content/drive/MyDrive/STUDI UMM/SEMESTER 7/PRAKTIKUM/UAP/MODEL/`.

### 1. CNN Base (Non-Pretrained)
- **Arsitektur**:
  - Conv2D (32 filters, 3x3, ReLU) + MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3, ReLU) + MaxPooling2D (2x2)
  - Conv2D (128 filters, 3x3, ReLU) + MaxPooling2D (2x2)
  - Flatten + Dense (512, ReLU) + BatchNormalization + Dropout (0.5)
  - Dense (5, softmax)
- **Learning Rate**: 0.001
- **Fitur Khusus**: Dibangun dari awal, tanpa bobot pre-trained. Cocok untuk baseline sederhana.
- **Ukuran Model**: Ringan (~2-3M parameter).

### 2. Pre-Trained MobileNetV2 (No Fine-Tuning)
- **Arsitektur**:
  - Base: MobileNetV2 (ImageNet weights, include_top=False, input (224x224x3))
  - Base trainable: False (frozen)
  - GlobalAveragePooling2D + BatchNormalization + Dropout (0.3)
  - Dense (512, ReLU) + BatchNormalization + Dropout (0.3)
  - Dense (256, ReLU) + Dropout (0.2)
  - Dense (5, softmax)
- **Learning Rate**: 1e-4
- **Fitur Khusus**: Transfer learning untuk efisiensi, ringan untuk mobile/edge devices.
- **Ukuran Model**: ~3-4M parameter (efisien).

### 3. Pre-Trained ResNet50 (No Fine-Tuning)
- **Arsitektur**:
  - Base: ResNet50 (ImageNet weights, include_top=False, input (224x224x3))
  - Base trainable: False (frozen)
  - GlobalAveragePooling2D + BatchNormalization + Dropout (0.3)
  - Dense (512, ReLU) + BatchNormalization + Dropout (0.3)
  - Dense (256, ReLU) + Dropout (0.2)
  - Dense (5, softmax)
- **Learning Rate**: 1e-4
- **Fitur Khusus**: Transfer learning dengan arsitektur residual yang kuat untuk akurasi tinggi.
- **Ukuran Model**: ~25M parameter (lebih berat, tapi akurat).

## Hasil Evaluasi dan Analisis Perbandingan

Evaluasi dilakukan pada test set (417 gambar) menggunakan metrik accuracy, loss, classification report, dan confusion matrix. Prediksi tambahan pada 50 sample random menunjukkan akurasi konsisten.

| Nama Model                  | Akurasi Test | Hasil Analisis |
|-----------------------------|--------------|----------------|
| **CNN Base (Non-Pretrained)** | 81.29%      | Model baseline ini menunjukkan performa sedang dengan F1-score macro avg 0.81. Recall rendah pada kelas 'tulip' (0.60) dan 'rose' (0.86), menandakan kesulitan membedakan variasi visual. Confusion matrix menunjukkan kesalahan cross-prediksi antara rose dan daisy. Cocok untuk resource terbatas, tapi butuh augmentasi lebih lanjut untuk imbalanced classes. Akurasi pada 50 sample: 84.0%. |
| **Pre-Trained MobileNetV2** | 98.56%      | Performa sangat baik dengan F1-score macro avg 0.99. Recall hampir sempurna (0.97-1.00), hanya sedikit kesalahan pada tulip/sunflower. Efisien dan cepat, ideal untuk deployment mobile. Confusion matrix bersih dengan diagonal dominan. Akurasi pada 50 sample: 100.0%. |
| **Pre-Trained ResNet50**    | 99.52%      | Model terbaik dengan F1-score macro avg 1.00 dan loss rendah (0.0281). Recall 0.99-1.00 di semua kelas, minim kesalahan. Kekuatan residual blocks membuatnya unggul dalam fitur ekstraksi kompleks. Confusion matrix hampir sempurna. Akurasi pada 50 sample: 100.0%. |

**Kesimpulan Perbandingan**: Model pre-trained (MobileNetV2 dan ResNet50) jauh lebih unggul daripada CNN base berkat bobot ImageNet, dengan ResNet50 sebagai pemenang. Pre-trained models mengurangi overfitting dan meningkatkan generalisasi. Untuk produksi, pilih ResNet50 jika akurasi prioritas, atau MobileNetV2 untuk efisiensi.

## Panduan Menjalankan Sistem Website Secara Lokal

Aplikasi web menggunakan Streamlit untuk interface sederhana: unggah gambar bunga dan dapatkan prediksi dari model terbaik (ResNet50).

### Prasyarat
- Python 3.8+ terinstall.
- VS Code dengan PowerShell terminal.
- Akses ke model yang disimpan (download dari MODEL.zip jika diperlukan).

### Langkah Instalasi dan Jalankan
1. **Install PDM (Package Manager)**:
   - Buka terminal PowerShell di VS Code.
   - Jalankan:  
     ```
     Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing | python -
     ```
   - Setelah instalasi, output akan menampilkan path seperti `C:\Users\<username>\AppData\Roaming\Python\Scripts`.
   - Copy path tersebut.
   - Buka **Advanced System Settings > Environment Variables > User Variables > New**, paste path, lalu restart terminal/VS Code.

2. **Inisialisasi Proyek**:
   - Di root folder `UAP/`, jalankan: `pdm init`.
   - Lengkapi prompt terminal (nama proyek: "Flower-Classification", dll.).

3. **Install Dependencies untuk Streamlit App**:
   - Pindah ke folder: `cd UAP/streamlit-app`.
   - Jalankan: `pip install -r requirements.txt` (ini akan install Streamlit, TensorFlow, dll.).

4. **Jalankan Aplikasi**:
   - Di folder `UAP/streamlit-app`, jalankan:  
     ```
     streamlit run app.py
     ```
   - Tunggu hingga browser terbuka otomatis di `http://localhost:8501` (atau buka manual).

5. **Penggunaan**:
   - Upload gambar bunga (format PNG/JPG/JPEG) dari kelas tulip, sunflower, rose, dandelion, atau daisy.
   - App akan memproses dan menampilkan prediksi kelas beserta confidence score menggunakan model CNN Base, MobileNetV2, atau ResNet50.
   - Contoh: Upload foto tulip → Prediksi: "tulip" dengan 99% confidence.

Jika error, pastikan model `.keras` ada di folder `streamlit-app/MODEL.zip` atau sesuaikan path di `app.py`.

## Contoh Hasil Klasifikasi

Berikut adalah contoh output klasifikasi dari aplikasi web menggunakan model ResNet50:

![Contoh Klasifikasi Tulip](images/tulip_example.jpg)

![Contoh Klasifikasi Sunflower](images/sunflower_example.jpg)

![Contoh Klasifikasi Rose](images/rose_example.jpg)

## Link Live Demo (Optional)
- [Streamlit Sharing](https://share.streamlit.io/) atau [Hugging Face Spaces](https://huggingface.co/spaces) (deploy manual belum dilakukan; hubungi developer untuk akses).

## Resources
- **Dataset**: [Google Drive - Flower Dataset](https://drive.google.com/drive/folders/1UoGVLAggyBttx7FxT_EUWbU6e0-v3z_Z?usp=sharing)
- **Model**: [Google Drive - Trained Models](https://drive.google.com/drive/folders/1F5HkKb1TZvRStKjq1hQeaWvo8UeyPN6n?usp=sharing) (Mengandung 3 model: cnn_base.keras, pretrained_mobilenetv2_no_finetune.keras, pretrained_resnet50_no_finetune.keras)
- **Google Colab Hasil Training**: [Colab Notebook](https://colab.research.google.com/drive/1owtefEu6OUjDkxaw6ww677f5RXtW1Aby?usp=sharing)