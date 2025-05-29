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

    # Tentukan dtype berdasarkan device
    if device == "cuda":
        # Gunakan float16 jika di GPU untuk efisiensi VRAM dan kecepatan
        torch_dtype = torch.float16
    else:
        # Gunakan float32 jika di CPU karena float16 tidak disarankan atau tidak didukung penuh
        torch_dtype = torch.float32
        # Pesan warning ini akan dipindahkan ke bagian Informasi Perangkat

    # Muat pipeline
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch_dtype
        )
        pipe = pipe.to(device)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()  # Hentikan eksekusi aplikasi jika gagal memuat model

    st.success("Model Stable Diffusion berhasil dimuat!")
    return pipe


# --- Judul dan Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Generasi Gambar AI - PXD Stable Diffusion - ",
    page_icon="✨",
    layout="wide",  # Menggunakan layout lebar untuk lebih banyak ruang
)

st.title("✨ Generasi Gambar dengan Stable Diffusion")
st.markdown("Ubah ide-ide kamu menjadi gambar menakjubkan!")

# --- Inisialisasi Model (ini akan dijalankan sekali) ---
with st.spinner("Menginisialisasi model Stable Diffusion..."):
    pipe = load_model()

# --- Inisialisasi Session State ---
# Ini penting untuk menyimpan gambar yang sudah digenerate
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "generated_prompt" not in st.session_state:
    st.session_state.generated_prompt = ""
if "generated_seed" not in st.session_state:
    st.session_state.generated_seed = None

# Inisialisasi status pilihan seed (Acak/Manual)
if "seed_choice" not in st.session_state:
    st.session_state.seed_choice = 'Acak' # Defaultnya "Acak"

# --- Antarmuka Pengguna untuk Prompt dan Parameter ---

# Bagian Prompt
st.header("🖊️ Masukkan Prompt kamu")
col_prompt, col_neg_prompt = st.columns(2)

with col_prompt:
    user_prompt = st.text_area(
        "**Prompt Positif (Apa yang ingin kamu lihat):**",
        "Vast mountain range at sunrise, mist in the valleys, clear alpine lake, golden hour light, majestic, landscape photography, sharp focus",
        height=150,
        help="Deskripsikan gambar yang kamu inginkan. Semakin detail, semakin baik!",
    )

with col_neg_prompt:
    user_negative_prompt = st.text_area(
        "**Prompt Negatif (Apa yang tidak ingin kamu lihat):**",
        "low quality, blurry, ugly, distorted, bad anatomy, deformed, text, watermark, extra fingers, malformed hands",
        height=150,
        help="Sebutkan hal-hal yang ingin kamu hindari di gambar (misal: kualitas buruk, distorsi).",
    )

st.markdown("---")

# Bagian Parameter
st.header("⚙️ Pengaturan Generasi")

# Informasi GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.success(f"**GPU:** {gpu_name} (CUDA Tersedia)")
else:
    st.warning("Tidak ada GPU terdeteksi. Menggunakan **CPU** untuk proses generasi gambar.")

 # --- Bagian Hires. fix ---
st.write("### ✨ Hires. fix (Perbaikan Resolusi Tinggi)")
enable_hires_fix = st.checkbox(
    "Aktifkan Hires. fix",
    help="Mengenerate gambar dengan dua langkah untuk kualitas lebih baik pada resolusi tinggi. Akan lebih lambat dan butuh VRAM lebih banyak."
)

if enable_hires_fix == True:
    # Kolom baru untuk resolusi
    col_res, col_hires, col_other = st.columns(3)
    
    with col_res:
        image_resolution = st.slider(
            "Resolusi Gambar (piksel)",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
            help="Resolusi gambar yang akan dihasilkan (persegi).  512 adalah optimal untuk Stable Diffusion v1.5.",
        )
        
        num_inference_steps = st.slider(
            "Jumlah Langkah Inferensi",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Jumlah langkah difusi. Angka lebih tinggi = detail lebih baik, tapi lebih lambat.",
        )

    with col_hires:
        if enable_hires_fix:
            hires_base_resolution = st.slider(
            "Resolusi Dasar (Langkah 1)",
            min_value=64,
            max_value=image_resolution, # Maksimal yang masih cukup aman untuk pass pertama
            value=int(image_resolution/2),
            step=64,
            help="Resolusi untuk generasi gambar pertama. Biasanya 512 untuk SD v1.5."
        )
        
        hires_denoising_strength = st.slider(
            "Denoising Strength (Langkah 2)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Seberapa banyak detail baru ditambahkan pada langkah kedua. Angka rendah = lebih mirip aslinya, angka tinggi = lebih banyak perubahan. (0.5 - 0.75 direkomendasikan)"
        )
        
        # Validasi untuk mencegah konfigurasi yang tidak masuk akal
        if hires_base_resolution > image_resolution:
            st.warning("Resolusi Dasar tidak boleh lebih besar dari Resolusi Gambar Akhir. Sesuaikan 'Resolusi Gambar' atau 'Resolusi Dasar'.")
            enable_hires_fix = False # Nonaktifkan Hires. fix jika konfigurasi tidak valid
        elif hires_base_resolution == image_resolution:
            st.info("Resolusi Dasar sama dengan Resolusi Gambar Akhir. Hires. fix mungkin tidak memberikan efek signifikan.")


    with col_other:
        guidance_scale = st.slider(
            "Skala Panduan (CFG Scale)",
            min_value=1.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
            help="Seberapa kuat model mengikuti prompt. Angka lebih tinggi = lebih patuh, tapi bisa kurang kreatif.",
        )
        
        # --- Bagian untuk Seed ---
        st.write("Pilihan Seed:") # Judul untuk pilihan seed
        
        # Buat sub-kolom untuk menempatkan tombol agar tertengah
        # Rasio [1, 1.5, 1.5, 1] akan memberikan dua kolom tengah yang lebih lebar untuk tombol
        btn_col1, btn_col_acak, btn_col_manual, btn_col4 = st.columns([1, 1.5, 1.5, 1])
    
        with btn_col_acak:
            # Tombol "Acak"
            # Tentukan type button berdasarkan state saat ini
            if st.button("Acak", use_container_width=True, type="primary" if st.session_state.seed_choice == 'Acak' else "secondary"):
                st.session_state.seed_choice = 'Acak'
                # Streamlit akan me-rerun secara otomatis jika session_state berubah dan widget lain tergantung padanya
    
        with btn_col_manual:
            # Tombol "Manual"
            if st.button("Manual", use_container_width=True, type="primary" if st.session_state.seed_choice == 'Manual' else "secondary"):
                st.session_state.seed_choice = 'Manual'
                # Streamlit akan me-rerun secara otomatis
    
        # Input seed manual akan muncul hanya jika 'Manual' dipilih
        if st.session_state.seed_choice == 'Manual':
            current_seed = st.number_input(
                "Masukkan Seed Manual",
                min_value=0, max_value=999999999, value=42, step=1,
                help="Gunakan seed yang sama untuk mendapatkan hasil yang sama dari prompt yang sama."
            )
        else:
            current_seed = None
        # --- Akhir bagian Seed ---

if enable_hires_fix == False:
    # Kolom baru untuk resolusi
    col_res, col_other = st.columns(2)
    
    with col_res:
        image_resolution = st.slider(
            "Resolusi Gambar (piksel)",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
            help="Resolusi gambar yang akan dihasilkan (persegi).  512 adalah optimal untuk Stable Diffusion v1.5.",
        )
        
        num_inference_steps = st.slider(
            "Jumlah Langkah Inferensi",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Jumlah langkah difusi. Angka lebih tinggi = detail lebih baik, tapi lebih lambat.",
        )

    with col_other:
        guidance_scale = st.slider(
            "Skala Panduan (CFG Scale)",
            min_value=1.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
            help="Seberapa kuat model mengikuti prompt. Angka lebih tinggi = lebih patuh, tapi bisa kurang kreatif.",
        )
        
        # --- Bagian untuk Seed ---
        st.write("Pilihan Seed:") # Judul untuk pilihan seed
        
        # Buat sub-kolom untuk menempatkan tombol agar tertengah
        # Rasio [1, 1.5, 1.5, 1] akan memberikan dua kolom tengah yang lebih lebar untuk tombol
        btn_col1, btn_col_acak, btn_col_manual, btn_col4 = st.columns([1, 1.5, 1.5, 1])
    
        with btn_col_acak:
            # Tombol "Acak"
            # Tentukan type button berdasarkan state saat ini
            if st.button("Acak", use_container_width=True, type="primary" if st.session_state.seed_choice == 'Acak' else "secondary"):
                st.session_state.seed_choice = 'Acak'
                # Streamlit akan me-rerun secara otomatis jika session_state berubah dan widget lain tergantung padanya
    
        with btn_col_manual:
            # Tombol "Manual"
            if st.button("Manual", use_container_width=True, type="primary" if st.session_state.seed_choice == 'Manual' else "secondary"):
                st.session_state.seed_choice = 'Manual'
                # Streamlit akan me-rerun secara otomatis
    
        # Input seed manual akan muncul hanya jika 'Manual' dipilih
        if st.session_state.seed_choice == 'Manual':
            current_seed = st.number_input(
                "Masukkan Seed Manual",
                min_value=0, max_value=999999999, value=42, step=1,
                help="Gunakan seed yang sama untuk mendapatkan hasil yang sama dari prompt yang sama."
            )
        else:
            current_seed = None

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
                final_seed = (
                    current_seed if st.session_state.seed_choice == "Manual" else random.randint(0, 999999999)
                )
                # Buat generator random untuk reproduktibilitas
                # Pastikan generator berada di device yang sama dengan pipe
                generator = torch.Generator(pipe.device).manual_seed(final_seed)

                if enable_hires_fix:
                    st.info(f"Langkah 1/2: Menggenerasi gambar dasar {hires_base_resolution}x{hires_base_resolution}...")
                    # Langkah 1: Generasi gambar dasar (Text-to-Image)
                    print('Generating..')
                    first_pass_image = pipe(
                        prompt=user_prompt,
                        negative_prompt=(user_negative_prompt if user_negative_prompt else None),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        width=hires_base_resolution,
                        height=hires_base_resolution,
                    ).images[0]

                    st.info(f"Langkah 2/2: Upscaling ke {image_resolution}x{image_resolution} dan memperbaiki detail...")
                    # Upscale gambar dasar menggunakan PIL
                    print('Upscaling..')
                    upscaled_image = first_pass_image.resize(
                        (image_resolution, image_resolution),
                        Image.LANCZOS # Metode upscaling yang lebih baik
                    )

                    # Langkah 2: Image-to-Image untuk perbaikan detail (Hires. fix)
                    final_image_output = pipe(
                        prompt=user_prompt,
                        image=upscaled_image, # Masukkan gambar hasil upscale sebagai input
                        negative_prompt=(user_negative_prompt if user_negative_prompt else None),
                        num_inference_steps=num_inference_steps, # Menggunakan langkah inferensi yang sama
                        guidance_scale=guidance_scale,
                        generator=generator,
                        strength=hires_denoising_strength, # Kekuatan denoising untuk img2img
                        width=image_resolution, # Pastikan width dan height sesuai dengan resolusi akhir
                        height=image_resolution,
                    ).images[0]
                else:
                    # Generasi langsung jika Hires. fix tidak aktif
                    print('\nGenerating..')
                    final_image_output = pipe(
                        prompt=user_prompt,
                        negative_prompt=(user_negative_prompt if user_negative_prompt else None),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        width=image_resolution,
                        height=image_resolution,
                    ).images[0]

                # --- Simpan gambar dan info ke session state ---
                st.session_state.generated_image = final_image_output
                st.session_state.generated_prompt = user_prompt
                st.session_state.generated_seed = final_seed
                # ---

                st.success("Gambar berhasil dihasilkan! Lihat hasilnya di bawah.")

            except torch.cuda.OutOfMemoryError:
                st.error("Terjadi kesalahan: GPU kehabisan memori (Out Of Memory).")
                st.warning("Coba kurangi 'Resolusi Gambar', 'Resolusi Dasar', 'Jumlah Langkah Inferensi', atau 'Skala Panduan'.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghasilkan gambar: {e}")
                st.warning(
                    "Beberapa penyebab umum: masalah dengan prompt. Coba kurangi 'Jumlah Langkah Inferensi' atau 'Skala Panduan', atau sederhanakan prompt kamu."
                )

# --- Tampilkan Gambar yang Dihasilkan (di luar blok if button) ---
# Ini akan memastikan gambar tetap ada bahkan setelah reruns
if st.session_state.generated_image:
    st.markdown("---")
    st.header("🖼️ Hasil Generasi")
    # Menggunakan use_container_width=True agar gambar menyesuaikan lebar kolom
    st.image(st.session_state.generated_image)

    # Tombol Download ditempatkan di sini
    buf = BytesIO()
    st.session_state.generated_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="💾 Unduh Gambar",
        data=byte_im,
        file_name=f"generated_image_{st.session_state.generated_seed}.png",
        mime="image/png",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("Dibuat dengan ❤️ oleh pianxd.")
