# ✨ PXD Image Generator

**PXD Image Generator** adalah aplikasi berbasis Streamlit yang memungkinkan Anda menghasilkan gambar AI dari deskripsi teks menggunakan model [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). Aplikasi ini dirancang untuk memberikan kontrol penuh kepada pengguna melalui pengaturan seperti prompt positif/negatif, jumlah langkah inferensi, skala panduan (CFG), dan seed acak atau manual.

---

## 🚀 Fitur Utama

- **Antarmuka Interaktif**: Masukkan prompt positif dan negatif untuk mengarahkan hasil gambar sesuai keinginan Anda.
- **Pengaturan Lanjutan**:
  - *Jumlah Langkah Inferensi*: Kontrol detail gambar dengan menyesuaikan jumlah langkah difusi.
  - *Skala Panduan (CFG Scale)*: Tentukan seberapa ketat model mengikuti prompt yang diberikan.
  - *Seed*: Pilih antara seed acak untuk variasi atau seed manual untuk reproduktibilitas.
- **Dukungan GPU/CPU Otomatis**: Aplikasi secara otomatis mendeteksi dan memanfaatkan GPU jika tersedia, atau beralih ke CPU dengan pengaturan yang sesuai.
- **Caching Model**: Menggunakan `@st.cache_resource` untuk memuat model sekali dan meningkatkan efisiensi.

---

## 🖼️ Contoh Prompt

**Prompt Positif**:
```
snow-capped mountain range at night reflecting a vibrant aurora borealis, long exposure, ethereal lighting, sense of wonder and tranquility
```

**Prompt Negatif**:
```
low quality, blurry, ugly, distorted, bad anatomy, deformed, text, watermark, extra fingers, malformed hands
```

---

## 🛠️ Instalasi & Menjalankan Aplikasi

1. **Kloning Repositori**:
   ```bash
   git clone https://github.com/Pixel4bit/pxd-image-generator.git
   cd pxd-image-generator
   ```

2. **Buat dan Aktifkan Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Unix atau MacOS
   venv\Scripts\activate     # Untuk Windows
   ```

3. **Instalasi Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## ⚙️ Persyaratan Sistem

- **Python**: Versi 3.8 atau lebih baru
- **Dependensi Utama**:
  - `streamlit`
  - `diffusers`
  - `transformers`
  - `accelerate`
  - `scipy`
  - `torch`
  - `Pillow`
  - `invisible_watermark`
- **Perangkat Keras**:
  - GPU dengan dukungan CUDA (opsional namun direkomendasikan untuk performa optimal)

---

## 📦 Struktur Proyek

```
pxd-image-generator/
├── streamlit_app.py       # Skrip utama aplikasi Streamlit
├── requirements.txt       # Daftar dependensi Python
└── README.md              # Dokumentasi proyek
```

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [Apache 2](LICENSE).

---

## 🙌 Kontribusi

Kontribusi sangat dihargai! Silakan buka *issue* atau *pull request* untuk perbaikan, fitur baru, atau diskusi lainnya.

---

## 📬 Kontak

Dikembangkan oleh [Pixel4bit (pianxd)](https://github.com/Pixel4bit). Untuk pertanyaan atau saran, silakan hubungi melalui GitHub.
