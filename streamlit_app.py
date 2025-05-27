import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random
import datetime # Meskipun tidak digunakan untuk seed dinamis, tetap ada jika Anda ingin mengaktifkan opsi timestamp
from io import BytesIO # Untuk tombol download gambar

# --- Konfigurasi Model ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# --- Cache Model untuk Efisiensi ---
# Menggunakan st.cache_resource agar model hanya dimuat sekali saat aplikasi dimulai
@st.cache_resource
def load_model():
    # Deteksi perangkat (GPU jika tersedia, jika tidak CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Memuat model Stable Diffusion ke: {device}. Ini mungkin memakan waktu beberapa detik...") # Pemberitahuan di UI
    
    # Muat pipeline dengan float16 untuk hemat VRAM jika menggunakan GPU
    # Fallback ke float32 jika di CPU atau jika float16 menyebabkan masalah
    try:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        pipe = pipe.to(device)
    except Exception as e:
        st.warning(f"Gagal memuat model dengan float16, mencoba dengan float32. Error: {e}")
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
        pipe = pipe.to(device)
    
    st.success("Model Stable Diffusion berhasil dimuat!")
    return pipe

# --- Judul dan Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Generasi Gambar AI",
    page_icon="✨",
    layout="wide", # Menggunakan layout lebar untuk lebih banyak ruang
    initial_sidebar_state="expanded" # Sidebar dibuka secara default
)

st.title("✨ Generasi Gambar AI dengan Stable Diffusion")
st.markdown("Ubah ide-ide Anda menjadi gambar menakjubkan!")

st.sidebar.header("Pengaturan Model")

# --- Inisialisasi Model (ini akan dijalankan sekali) ---
with st.spinner("Menginisialisasi model Stable Diffusion..."):
    pipe = load_model()

st.sidebar.success("Model siap digunakan!")

# --- Antarmuka Pengguna untuk Prompt dan Parameter ---

# Bagian Prompt
st.header("🖊️ Masukkan Prompt Anda")
col_prompt, col_neg_prompt = st.columns(2)

with col_prompt:
    user_prompt = st.text_area(
        "**Prompt Positif (Apa yang ingin Anda lihat):**",
        "a majestic fantasy landscape, digital art, highly detailed, beautiful lighting, trending on artstation",
        height=150,
        help="Deskripsikan gambar yang Anda inginkan. Semakin detail, semakin baik!"
    )

with col_neg_prompt:
    user_negative_prompt = st.text_area(
        "**Prompt Negatif (Apa yang tidak ingin Anda lihat):**",
        "low quality, blurry, ugly, distorted, bad anatomy, deformed, text, watermark, extra fingers, malformed hands",
        height=150,
        help="Sebutkan hal-hal yang ingin Anda hindari di gambar (misal: kualitas buruk, distorsi)."
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
        # Gunakan seed acak baru setiap kali tombol generate ditekan
        # Ini akan diinisialisasi ulang di dalam fungsi generate
        current_seed = None 
        st.markdown("_Seed akan dihasilkan secara acak saat Anda menekan Generate._")
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
        with st.spinner("Memproses gambar Anda... Ini mungkin memakan waktu beberapa detik."):
            try:
                # Dapatkan seed yang sebenarnya
                final_seed = current_seed if seed_option == 'Manual' else random.randint(0, 999999999)
                st.info(f"Menggunakan seed: `{final_seed}`")

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

                st.success("Gambar berhasil dihasilkan!")
                st.image(image_output, caption=f"Gambar dari Prompt: '{user_prompt}' (Seed: {final_seed})", use_column_width=True)

                # Tombol Download
                buf = BytesIO()
                image_output.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="💾 Unduh Gambar",
                    data=byte_im,
                    file_name=f"generated_image_{final_seed}.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghasilkan gambar: {e}")
                st.warning("Beberapa penyebab umum: GPU kehabisan memori, atau masalah dengan prompt. Coba kurangi 'Jumlah Langkah Inferensi' atau 'Skala Panduan', atau sederhanakan prompt Anda.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ dan Streamlit.")