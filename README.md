# ✨ PXD Stable Diffusion Image-Generator

**PXD Image Generator** is a Streamlit-based application that allows you to generate AI images from text descriptions using the [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) model. This app is designed to give users full control through options like positive/negative prompts, number of inference steps, guidance scale (CFG), and random or manual seed selection.

---

## 🚀 Key Features

- **Interactive Interface**: Enter positive and negative prompts to guide image results according to your preferences.
- **Advanced Settings**:
  - *Number of Inference Steps*: Control image detail by adjusting the number of diffusion steps.
  - *Guidance Scale (CFG Scale)*: Set how closely the model follows the given prompt.
  - *Seed*: Choose between a random seed for variation or a manual seed for reproducibility.
- **Automatic GPU/CPU Support**: Automatically detects and utilizes GPU if available, or falls back to CPU with appropriate settings.
- **Model Caching**: Uses `@st.cache_resource` to load the model once and improve efficiency.

---

## 🖼️ Example Prompts

**Positive Prompt**:
```
snow-capped mountain range at night reflecting a vibrant aurora borealis, long exposure, ethereal lighting, sense of wonder and tranquility
```

**Negative Prompt**:
```
low quality, blurry, ugly, distorted, bad anatomy, deformed, text, watermark, extra fingers, malformed hands
```

---

## 🛠️ Installation & Running the App

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pixel4bit/pxd-image-generator.git
   cd pxd-image-generator
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Unix or MacOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## ⚙️ System Requirements

- **Python**: Version 3.8 or later
- **Core Dependencies**:
  - `streamlit`
  - `diffusers`
  - `transformers`
  - `accelerate`
  - `scipy`
  - `torch`
  - `Pillow`
  - `invisible_watermark`
- **Hardware**:
  - GPU with CUDA support (optional but recommended for optimal performance)

---

## 📦 Project Structure

```
pxd-image-generator/
├── streamlit_app.py       # Main Streamlit app script
├── requirements.txt       # Python dependencies list
└── README.md              # Project documentation
```

---

## 📄 License

This project is licensed under the [Apache 2.0](LICENSE) License.

---

## 🙌 Contributions

Contributions are welcome! Feel free to open an *issue* or *pull request* for improvements, new features, or discussions.

---

## 📬 Contact

Developed by [Pixel4bit (pianxd)](https://github.com/Pixel4bit). For questions or suggestions, feel free to reach out via GitHub.
