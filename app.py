import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="LeafCare – Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background-color: #f5f0e8;
    background-image:
        radial-gradient(circle at 10% 20%, rgba(134,176,116,0.15) 0%, transparent 50%),
        radial-gradient(circle at 90% 80%, rgba(212,175,130,0.15) 0%, transparent 50%);
    font-family: 'DM Sans', sans-serif;
}

/* Hide streamlit header */
header[data-testid="stHeader"] { background: transparent; }

/* ---- NAVBAR ---- */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 36px;
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1.5px solid rgba(134,176,116,0.25);
    margin-bottom: 36px;
    box-shadow: 0 4px 24px rgba(134,176,116,0.12);
}

.logo {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: #3d6b35;
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-dot {
    width: 10px; height: 10px;
    background: #86b074;
    border-radius: 50%;
    display: inline-block;
}

.nav-badge {
    background: #e8f3e4;
    color: #3d6b35;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 16px;
    border-radius: 999px;
    border: 1px solid rgba(134,176,116,0.3);
}

/* ---- HERO ---- */
.hero {
    text-align: center;
    padding: 20px 20px 36px;
}

.hero-eyebrow {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #86b074;
    margin-bottom: 14px;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 52px;
    color: #2c4a26;
    line-height: 1.1;
    margin-bottom: 14px;
}

.hero-title em {
    font-style: italic;
    color: #5a8c4e;
}

.hero-sub {
    font-size: 17px;
    color: #6b7c65;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ---- DECORATIVE DIVIDER ---- */
.divider {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin: 8px 0 32px;
    color: #b5c9b0;
    font-size: 18px;
}

.divider-line {
    height: 1px;
    width: 60px;
    background: linear-gradient(to right, transparent, #b5c9b0);
}

.divider-line.right {
    background: linear-gradient(to left, transparent, #b5c9b0);
}

/* ---- UPLOAD CARD ---- */
.upload-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(16px);
    border-radius: 28px;
    border: 1.5px solid rgba(134,176,116,0.2);
    padding: 32px;
    box-shadow: 0 8px 32px rgba(93,139,82,0.08), 0 2px 8px rgba(0,0,0,0.04);
}

.upload-label {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: #2c4a26;
    margin-bottom: 6px;
}

.upload-hint {
    font-size: 13px;
    color: #8fa889;
    margin-bottom: 18px;
}

/* ---- RESULT CARDS ---- */
.result-healthy {
    background: linear-gradient(135deg, #eaf4e6 0%, #d4ead0 100%);
    border: 1.5px solid #a8d4a2;
    border-radius: 20px;
    padding: 24px 28px;
    margin-top: 20px;
}

.result-disease {
    background: linear-gradient(135deg, #fdf0ec 0%, #f9ddd5 100%);
    border: 1.5px solid #f0b8a8;
    border-radius: 20px;
    padding: 24px 28px;
    margin-top: 20px;
}

.result-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.result-healthy .result-label { color: #4a8c42; }
.result-disease .result-label { color: #c05a3a; }

.result-name {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    margin-bottom: 8px;
}

.result-healthy .result-name { color: #2d5c28; }
.result-disease .result-name { color: #8c3520; }

.result-desc {
    font-size: 14px;
    line-height: 1.6;
}

.result-healthy .result-desc { color: #5a8c55; }
.result-disease .result-desc { color: #a04832; }

/* ---- CONFIDENCE BAR ---- */
.conf-section {
    margin-top: 20px;
    background: rgba(255,255,255,0.6);
    border-radius: 16px;
    padding: 18px 22px;
    border: 1px solid rgba(134,176,116,0.15);
}

.conf-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #86b074;
    margin-bottom: 10px;
}

/* ---- INFO CARDS RIGHT SIDE ---- */
.info-card {
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(12px);
    border-radius: 22px;
    border: 1.5px solid rgba(134,176,116,0.18);
    padding: 24px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(93,139,82,0.06);
    transition: transform 0.2s;
}

.info-card:hover { transform: translateY(-4px); }

.info-icon {
    font-size: 32px;
    margin-bottom: 10px;
}

.info-title {
    font-family: 'DM Serif Display', serif;
    font-size: 18px;
    color: #2c4a26;
    margin-bottom: 6px;
}

.info-text {
    font-size: 14px;
    color: #6b7c65;
    line-height: 1.6;
}

/* ---- PLANTS PILLS ---- */
.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.pill {
    background: #e8f3e4;
    color: #3d6b35;
    font-size: 12px;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 999px;
    border: 1px solid rgba(134,176,116,0.3);
}

/* ---- FOOTER ---- */
.footer {
    text-align: center;
    margin-top: 56px;
    padding: 24px;
    font-size: 14px;
    color: #8fa889;
    border-top: 1px solid rgba(134,176,116,0.15);
}

.footer span { color: #5a8c4e; font-weight: 500; }

/* ---- STREAMLIT OVERRIDES ---- */
.stFileUploader > div {
    background: #f0f7ee !important;
    border: 2px dashed #a8c9a2 !important;
    border-radius: 16px !important;
}

.stProgress > div > div > div {
    background: linear-gradient(to right, #86b074, #4a8c42) !important;
    border-radius: 999px !important;
}

.stSpinner { color: #5a8c4e !important; }

div[data-testid="stImage"] img {
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
}

</style>
""", unsafe_allow_html=True)


# ---- MODEL ----
model = tf.keras.models.load_model("plant_disease_model.keras")

class_names = [
    "Pepper Bacterial Spot", "Pepper Healthy",
    "Potato Early Blight", "Potato Healthy", "Potato Late Blight",
    "Tomato Target Spot", "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus", "Tomato Bacterial Spot",
    "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites"
]


# ---- NAVBAR ----
st.markdown("""
<div class="navbar">
    <div class="logo">
        <span>🌿</span> LeafCare
    </div>
    <div class="nav-badge">🤖 AI-Powered Analysis</div>
</div>
""", unsafe_allow_html=True)


# ---- HERO ----
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🌱 Plant Health Intelligence</div>
    <div class="hero-title">Is your plant <em>okay?</em></div>
    <div class="hero-sub">
        Upload a photo of any leaf and our AI will gently
        tell you what's going on — in seconds.
    </div>
</div>
<div class="divider">
    <div class="divider-line"></div>
    🌸
    <div class="divider-line right"></div>
</div>
""", unsafe_allow_html=True)


# ---- MAIN LAYOUT ----
left_col, right_col = st.columns([1.3, 1], gap="large")


with left_col:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<div class="upload-label">📷 Upload a leaf photo</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">JPG, JPEG or PNG · Best with clear, well-lit leaves</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded leaf 🍃", use_container_width=True)

        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Examining your leaf..."):
            prediction = model.predict(img_array)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result = class_names[predicted_class]

        if "Healthy" in result:
            st.markdown(f"""
            <div class="result-healthy">
                <div class="result-label">✅ All Clear</div>
                <div class="result-name">{result}</div>
                <div class="result-desc">
                    Your plant looks happy and thriving! Keep up the good care
                    with proper watering, sunlight and love. 🌞
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-disease">
                <div class="result-label">⚠️ Disease Detected</div>
                <div class="result-name">{result}</div>
                <div class="result-desc">
                    We spotted signs of this condition. Consider consulting
                    a plant care guide or an agronomist for treatment options. 🌿
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="conf-section">', unsafe_allow_html=True)
        st.markdown('<div class="conf-label">📊 Confidence Score</div>', unsafe_allow_html=True)
        st.progress(int(confidence))
        st.markdown(f"<p style='font-size:20px; font-weight:600; color:#3d6b35; margin-top:6px;'>{confidence:.1f}%</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


with right_col:

    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🧠</div>
        <div class="info-title">Deep Learning Model</div>
        <div class="info-text">
            Powered by MobileNetV2 with transfer learning,
            trained on the PlantVillage dataset for
            high accuracy leaf classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-icon">⚡</div>
        <div class="info-title">Instant Results</div>
        <div class="info-text">
            Analysis completes in seconds using an
            optimized TensorFlow pipeline — no waiting,
            no guessing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🌱</div>
        <div class="info-title">Supported Plants</div>
        <div class="info-text">Currently detecting diseases across:</div>
        <div class="pills">
            <span class="pill">🍅 Tomato</span>
            <span class="pill">🥔 Potato</span>
            <span class="pill">🫑 Pepper</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-icon">💡</div>
        <div class="info-title">Tips for best results</div>
        <div class="info-text">
            📸 Use natural daylight<br>
            🍃 Focus on a single leaf<br>
            🔍 Avoid blurry or dark photos<br>
            🌿 Show the full leaf surface
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---- FOOTER ----
st.markdown("""
<div class="footer">
    Made with 🌿 &amp; ❤️ using <span>TensorFlow</span> · <span>Keras</span> · <span>Streamlit</span>
</div>
""", unsafe_allow_html=True)