import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import datetime
from io import BytesIO

# --- Konfigurasi Model ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# --- Cache Model untuk Efisiensi ---
@st.cache_resource
def load_model():
    # Deteksi perangkat (GPU jika tersedia, jika tidak CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Memuat model Stable Diffusion. Ini mungkin memakan waktu beberapa detik...") # Pemberitahuan di UI

    # Tentukan dtype berdasarkan device
    if device == "cuda":
        # Gunakan float16 jika di GPU untuk efisiensi VRAM dan kecepatan
        torch_dtype = torch.float16
    else:
        # Gunakan float32 jika di CPU karena float16 tidak disarankan atau tidak didukung penuh
        torch_dtype = torch.float32
        st.warning("GPU tidak terdeteksi. Menggunakan CPU dengan float32. Generasi gambar akan jauh lebih lambat.")

    # Muat pipeline
    try:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop() # Hentikan eksekusi aplikasi jika gagal memuat model

    st.success("Model Stable Diffusion berhasil dimuat!")
    return pipe

# --- Judul dan Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Generasi Gambar AI",
    page_icon="✨",
    layout="wide", # Menggunakan layout lebar untuk lebih banyak ruang
)

st.title("✨ Generasi Gambar dengan Stable Diffusion")
st.markdown("Ubah ide-ide kamu menjadi gambar menakjubkan!")

# --- Inisialisasi Model (ini akan dijalankan sekali) ---
with st.spinner("Menginisialisasi model Stable Diffusion..."):
    pipe = load_model()

# --- Inisialisasi Session State ---
# Ini penting untuk menyimpan gambar yang sudah digenerate
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'generated_prompt' not in st.session_state:
    st.session_state.generated_prompt = ""
if 'generated_seed' not in st.session_state:
    st.session_state.generated_seed = None

# --- Antarmuka Pengguna untuk Prompt dan Parameter ---

# Bagian Prompt
st.header("🖊️ Masukkan Prompt kamu")
col_prompt, col_neg_prompt = st.columns(2)

with col_prompt:
    user_prompt = st.text_area(
        "**Prompt Positif (Apa yang ingin kamu lihat):**",
        "snow-capped mountain range at night reflecting a vibrant aurora borealis, long exposure, ethereal lighting, sense of wonder and tranquility",
        height=150,
        help="Deskripsikan gambar yang kamu inginkan. Semakin detail, semakin baik!"
    )

with col_neg_prompt:
    user_negative_prompt = st.text_area(
        "**Prompt Negatif (Apa yang tidak ingin kamu lihat):**",
        "low quality, blurry, ugly, distorted, bad anatomy, deformed, text, watermark, extra fingers, malformed hands",
        height=150,
        help="Sebutkan hal-hal yang ingin kamu hindari di gambar (misal: kualitas buruk, distorsi)."
    )

st.markdown("---")

# Bagian Parameter
st.header("⚙️ Pengaturan Generasi")

col1, col2, col3 = st.columns(3)

with col1:
    num_inference_steps = st.slider(
        "Jumlah Langkah Inferensi",
        min_value=10, max_value=100, value=30, step=5,
        help="Jumlah langkah difusi. Angka lebih tinggi = detail lebih baik, tapi lebih lambat."
    )
with col2:
    guidance_scale = st.slider(
        "Skala Panduan (CFG Scale)",
        min_value=1.0, max_value=20.0, value=8.5, step=0.5,
        help="Seberapa kuat model mengikuti prompt. Angka lebih tinggi = lebih patuh, tapi bisa kurang kreatif."
    )
with col3:
    # Memilih seed: dinamis atau input manual
    seed_option = st.radio(
        "Pilih Seed",
        ('Acak', 'Manual'),
        horizontal=True,
        help="Seed mengontrol keacakan gambar. 'Acak' akan menghasilkan gambar unik setiap kali."
    )
    if seed_option == 'Acak':
        current_seed = None
    else:
        current_seed = st.number_input(
            "Masukkan Seed Manual",
            min_value=0, max_value=999999999, value=42, step=1,
            help="Gunakan seed yang sama untuk mendapatkan hasil yang sama dari prompt yang sama."
        )

st.markdown("---")

# Tombol Generate
if st.button("✨ Generate Gambar!", use_container_width=True, type="primary"):
    if not user_prompt:
        st.warning("Prompt positif tidak boleh kosong!")
    else:
        # Menampilkan pesan loading
        with st.spinner("Memproses gambar kamu..."):
            try:
                # Dapatkan seed yang sebenarnya
                final_seed = current_seed if seed_option == 'Manual' else random.randint(0, 999999999)

                # Buat generator random untuk reproduktibilitas
                # Pastikan generator berada di device yang sama dengan pipe
                generator = torch.Generator(pipe.device).manual_seed(final_seed)

                # Lakukan inferensi
                with torch.no_grad(): # Matikan autograd untuk menghemat memori saat inferensi
                    image_output = pipe(
                        prompt=user_prompt,
                        negative_prompt=user_negative_prompt if user_negative_prompt else None, # Kirim None jika kosong
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator
                    ).images[0]

                # --- Simpan gambar dan info ke session state ---
                st.session_state.generated_image = image_output
                st.session_state.generated_prompt = user_prompt
                st.session_state.generated_seed = final_seed
                # ---

                st.success("Gambar berhasil dihasilkan! Lihat hasilnya di bawah.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghasilkan gambar: {e}")
                st.warning("Beberapa penyebab umum: GPU kehabisan memori, atau masalah dengan prompt. Coba kurangi 'Jumlah Langkah Inferensi' atau 'Skala Panduan', atau sederhanakan prompt kamu.")

# --- Tampilkan Gambar yang Dihasilkan (di luar blok if button) ---
# Ini akan memastikan gambar tetap ada bahkan setelah reruns
if st.session_state.generated_image:
    st.markdown("---")
    st.header("🖼️ Hasil Generasi")
    st.image(st.session_state.generated_image, use_container_width=True)

    # Tombol Download ditempatkan di sini
    buf = BytesIO()
    st.session_state.generated_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="💾 Unduh Gambar",
        data=byte_im,
        file_name=f"generated_image_{st.session_state.generated_seed}.png",
        mime="image/png",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("Dibuat dengan ❤️ dan Streamlit.")