import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MammoSense | Breast Ultrasound AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "Makky07/MammoSense-breast-ultrasound"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   GLOBAL
========================================================== */

.stApp {
    background: #F7F9FC;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ==========================================================
   REMOVE STREAMLIT CLUTTER
========================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ==========================================================
   HERO
========================================================== */

.hero {
    background:
        radial-gradient(
            circle at 90% 20%,
            rgba(96,165,250,0.30),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #0F172A,
            #172554 55%,
            #1D4ED8
        );

    border-radius: 28px;

    padding: 42px 45px;

    margin-bottom: 30px;

    color: white;

    box-shadow:
        0 20px 60px rgba(15,23,42,0.15);
}

.hero-top {
    display: flex;
    align-items: center;
    gap: 15px;
}

.logo {
    width: 58px;
    height: 58px;

    border-radius: 16px;

    background: rgba(255,255,255,0.12);

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 30px;

    border: 1px solid rgba(255,255,255,0.18);
}

.hero h1 {
    font-size: 44px;
    font-weight: 750;

    letter-spacing: -1.5px;

    margin: 0;
}

.hero-subtitle {
    font-size: 17px;

    margin-top: 16px;

    color: #DBEAFE;

    max-width: 700px;

    line-height: 1.7;
}

.hero-badge {
    display: inline-block;

    margin-top: 22px;

    padding: 7px 13px;

    border-radius: 100px;

    background: rgba(255,255,255,0.10);

    border: 1px solid rgba(255,255,255,0.16);

    font-size: 13px;

    color: #E0F2FE;
}


/* ==========================================================
   SECTION HEADINGS
========================================================== */

.section-title {
    font-size: 26px;

    font-weight: 720;

    color: #0F172A;

    letter-spacing: -0.5px;

    margin-top: 20px;

    margin-bottom: 6px;
}

.section-description {
    color: #64748B;

    font-size: 15px;

    margin-bottom: 22px;
}


/* ==========================================================
   CARDS
========================================================== */

.card {
    background: white;

    border: 1px solid #E5EAF1;

    border-radius: 20px;

    padding: 25px;

    box-shadow:
        0 8px 30px rgba(15,23,42,0.04);
}

.card-title {
    font-size: 16px;

    font-weight: 700;

    color: #0F172A;

    margin-bottom: 8px;
}


/* ==========================================================
   UPLOAD AREA
========================================================== */

.upload-card {
    background: white;

    border: 1.5px dashed #CBD5E1;

    border-radius: 22px;

    padding: 35px 25px;

    text-align: center;

    transition: 0.2s ease;
}

.upload-icon {
    font-size: 42px;

    margin-bottom: 10px;
}

.upload-title {
    font-size: 20px;

    font-weight: 700;

    color: #0F172A;
}

.upload-description {
    color: #64748B;

    font-size: 14px;

    margin-top: 6px;
}


/* ==========================================================
   IMAGE
========================================================== */

.image-card {
    background: white;

    border-radius: 22px;

    padding: 15px;

    border: 1px solid #E5EAF1;

    box-shadow:
        0 8px 30px rgba(15,23,42,0.05);
}


/* ==========================================================
   RESULT
========================================================== */

.result-card {
    background: white;

    border-radius: 22px;

    padding: 28px;

    border: 1px solid #E5EAF1;

    box-shadow:
        0 8px 30px rgba(15,23,42,0.05);
}

.result-label {
    color: #64748B;

    font-size: 13px;

    text-transform: uppercase;

    letter-spacing: 1px;

    font-weight: 700;
}

.result-value {
    font-size: 38px;

    font-weight: 780;

    color: #0F172A;

    margin-top: 5px;

    letter-spacing: -1px;
}

.result-confidence {
    color: #2563EB;

    font-weight: 700;

    font-size: 16px;
}


/* ==========================================================
   STATUS
========================================================== */

.status {
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 7px 12px;

    border-radius: 100px;

    background: #ECFDF5;

    color: #047857;

    font-size: 13px;

    font-weight: 700;
}

.status-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #10B981;
}


/* ==========================================================
   DISCLAIMER
========================================================== */

.disclaimer {
    background: #FFF7ED;

    border: 1px solid #FED7AA;

    border-radius: 18px;

    padding: 22px;

    color: #7C2D12;

    line-height: 1.65;

    font-size: 14px;
}


/* ==========================================================
   MODEL INFO
========================================================== */

.model-info {
    background: #F8FAFC;

    border-radius: 16px;

    padding: 18px;

    border: 1px solid #E2E8F0;
}

.model-name {
    font-weight: 750;

    color: #0F172A;

    font-size: 16px;
}

.model-detail {
    color: #64748B;

    font-size: 13px;

    margin-top: 5px;
}


/* ==========================================================
   FOOTER
========================================================== */

.footer {
    text-align: center;

    color: #94A3B8;

    font-size: 12px;

    margin-top: 40px;
}


/* ==========================================================
   MOBILE
========================================================== */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 28px 25px;
        border-radius: 22px;
    }

    .hero h1 {
        font-size: 34px;
    }

    .hero-subtitle {
        font-size: 15px;
    }

    .result-value {
        font-size: 32px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE
# EXACTLY MATCHES TRAINING
# ============================================================

class BUSIViT(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )

        self.head = nn.Sequential(

            nn.Linear(
                self.backbone.embed_dim,
                1024
            ),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(
                1024,
                512
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

@st.cache_resource
def download_model():

    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type="model"
    )

    model_files = [
        f for f in files
        if f.lower().endswith(".pt")
    ]

    if not model_files:

        raise FileNotFoundError(
            "No .pt model file was found in the "
            "MammoSense Hugging Face repository."
        )

    model_filename = model_files[0]

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=model_filename,
        repo_type="model"
    )

    return model_path, model_filename


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path, model_filename = download_model()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model = BUSIViT(
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    return (
        model,
        checkpoint,
        model_filename
    )


# ============================================================
# PREPROCESSING
# ============================================================

transform = T.Compose([

    T.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    T.ToTensor(),

    T.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# PREDICTION
# ============================================================

def predict(image, model):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    index = torch.argmax(
        probabilities
    ).item()

    return (
        CLASS_NAMES[index],
        probabilities.cpu().numpy()
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-top">

<div class="logo">
🩺
</div>

<div>

<h1>MammoSense</h1>

</div>

</div>

<div class="hero-subtitle">

Intelligent breast ultrasound analysis powered by
a Vision Transformer trained for three-class
classification.

</div>

<div class="hero-badge">

RESEARCH AI • BUSI • VISION TRANSFORMER

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL STATUS
# ============================================================

try:

    with st.spinner(
        "Initializing MammoSense AI..."
    ):

        model, checkpoint, model_filename = load_model()

    st.markdown(
        """
        <div class="status">
        <span class="status-dot"></span>
        AI model online
        </div>
        """,
        unsafe_allow_html=True
    )

except Exception as e:

    st.error(
        "MammoSense could not initialize."
    )

    with st.expander(
        "Technical details"
    ):

        st.code(
            str(e)
        )

    st.stop()


# ============================================================
# INTRO
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Breast Ultrasound Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Upload an ultrasound image to generate an '
    'AI classification and probability profile.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ],
    label_visibility="collapsed"
)


if uploaded_file is None:

    st.markdown("""
    <div class="upload-card">

    <div class="upload-icon">
    🖼️
    </div>

    <div class="upload-title">
    Drop your ultrasound image here
    </div>

    <div class="upload-description">
    JPG, JPEG, PNG, BMP, TIFF • Recommended input quality
    </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    prediction, probabilities = predict(
        image,
        model
    )

    confidence = (
        float(probabilities.max()) * 100
    )


    # ========================================================
    # IMAGE + RESULT
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    image_col, result_col = st.columns(
        [1.05, 0.95],
        gap="large"
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_col:

        st.markdown(
            '<div class="card-title">'
            'Ultrasound image'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result_col:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">'
            'AI classification'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">'
            f'{prediction}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-confidence">'
            f'{confidence:.2f}% model probability'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.write(
            "Probability distribution"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                float(probabilities[i]) * 100
            )

            st.markdown(
                f"**{class_name}**"
            )

            st.progress(
                min(
                    probability / 100,
                    1.0
                )
            )

            st.caption(
                f"{probability:.2f}%"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Prediction overview'
        '</div>',
        unsafe_allow_html=True
    )

    overview1, overview2, overview3 = st.columns(3)


    with overview1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-title">'
            'Top prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"### {prediction}"
        )

        st.caption(
            "Highest probability class"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with overview2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-title">'
            'Model confidence'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"### {confidence:.1f}%"
        )

        st.caption(
            "Probability assigned to top class"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with overview3:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-title">'
            'Image input'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "224 × 224"
        )

        st.caption(
            "Model inference resolution"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MODEL DETAILS
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

with st.expander(
    "MammoSense model details"
):

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            "**Architecture**"
        )

        st.write(
            "ViT-Small Patch16-224"
        )

    with c2:

        st.markdown(
            "**Training dataset**"
        )

        st.write(
            "BUSI"
        )

    with c3:

        st.markdown(
            "**Classes**"
        )

        st.write(
            "Normal • Benign • Malignant"
        )

    st.divider()

    st.write(
        f"**Model file:** `{model_filename}`"
    )

    st.write(
        "**Framework:** PyTorch + timm"
    )

    st.write(
        "**Input preprocessing:** "
        "Resize → Tensor → ImageNet normalization"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer">

<strong>⚠️ Research & Medical Disclaimer</strong>

<br><br>

MammoSense is an experimental research prototype
for AI-assisted breast ultrasound image classification.
It is <strong>not a medical device</strong> and must not
be used as a substitute for professional medical
evaluation or diagnosis.

<br><br>

The model can produce incorrect predictions and may
perform differently on images outside its training
distribution. A qualified healthcare professional
should interpret ultrasound findings and make any
clinical decisions.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

MammoSense · Breast Ultrasound AI Research Prototype

<br>

ViT-Small Patch16-224 · BUSI · PyTorch

</div>
""", unsafe_allow_html=True)
