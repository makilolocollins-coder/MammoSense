import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files
import numpy as np
import cv2


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
# CONFIG
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

.stApp {
    background: #F7F9FC;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* HERO */

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
    max-width: 720px;
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

/* HEADINGS */

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

/* CARDS */

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

/* STATUS */

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

/* RESULT */

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
}

.result-confidence {
    color: #2563EB;
    font-weight: 700;
    font-size: 16px;
}

/* EXPLAINABILITY */

.explain-card {
    background: white;

    border: 1px solid #E5EAF1;

    border-radius: 22px;

    padding: 26px;

    box-shadow:
        0 8px 30px rgba(15,23,42,0.04);
}

.explain-title {
    font-size: 22px;
    font-weight: 750;
    color: #0F172A;
}

.explain-description {
    color: #64748B;
    font-size: 14px;
    line-height: 1.6;
}

/* DISCLAIMER */

.disclaimer {
    background: #FFF7ED;

    border: 1px solid #FED7AA;

    border-radius: 18px;

    padding: 22px;

    color: #7C2D12;

    line-height: 1.65;

    font-size: 14px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    margin-top: 40px;
}

/* MOBILE */

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
# MODEL
# EXACT ARCHITECTURE USED DURING TRAINING
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
# SAME AS TRAINING EVALUATION
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
# VIT GRAD-CAM
# ============================================================

class ViTGradCAM:

    def __init__(self, model):

        self.model = model

        self.activations = None
        self.gradients = None

        # Last transformer block
        self.target_layer = (
            model.backbone.blocks[-1].norm1
        )

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        input_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        score = output[
            0,
            class_index
        ]

        score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):

            raise RuntimeError(
                "Grad-CAM activations or gradients "
                "could not be captured."
            )

        activations = self.activations
        gradients = self.gradients

        # ----------------------------------------------------
        # ViT output can be:
        #
        # [batch, tokens, channels]
        #
        # We remove CLS token.
        # ----------------------------------------------------

        if activations.ndim == 3:

            activations = activations[:, 1:, :]

            gradients = gradients[:, 1:, :]

            tokens = activations.shape[1]

            grid_size = int(
                np.sqrt(tokens)
            )

            if (
                grid_size * grid_size
                != tokens
            ):

                raise RuntimeError(
                    f"Unexpected ViT token count: {tokens}"
                )

            activations = activations.reshape(
                1,
                grid_size,
                grid_size,
                -1
            )

            gradients = gradients.reshape(
                1,
                grid_size,
                grid_size,
                -1
            )

            # Channel weighting
            weights = gradients.mean(
                dim=(1, 2),
                keepdim=True
            )

            cam = (
                activations * weights
            ).sum(dim=-1)

        else:

            raise RuntimeError(
                "Unexpected ViT activation shape."
            )

        cam = torch.relu(
            cam
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize
        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam

    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    image,
    cam
):

    original = np.array(
        image.convert("RGB")
    )

    height, width = (
        original.shape[:2]
    )

    cam_resized = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        255 * cam_resized
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        0.55 * heatmap
        +
        0.45 * original
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return (
        heatmap,
        overlay
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    image,
    model
):

    image_rgb = image.convert(
        "RGB"
    )

    tensor = transform(
        image_rgb
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor.requires_grad_(True)

    # Grad-CAM requires gradients
    model.zero_grad()

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # Generate Grad-CAM
    gradcam = ViTGradCAM(
        model
    )

    cam = gradcam.generate(
        tensor,
        predicted_index
    )

    gradcam.close()

    return (
        predicted_class,
        predicted_index,
        probabilities.detach().cpu().numpy(),
        cam
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

AI-assisted breast ultrasound analysis powered by
a Vision Transformer trained on the BUSI dataset.

</div>

<div class="hero-badge">

RESEARCH AI &nbsp;•&nbsp; BUSI &nbsp;•&nbsp;
ViT-SMALL &nbsp;•&nbsp; EXPLAINABLE AI

</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Initializing MammoSense AI..."
    ):

        model, checkpoint, model_filename = (
            load_model()
        )

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
    'AI classification and visualize the regions '
    'that influenced the model prediction.'
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
    <div class="card">

    <div style="
        text-align:center;
        padding:30px;
    ">

    <div style="
        font-size:45px;
        margin-bottom:12px;
    ">
    🖼️
    </div>

    <div style="
        font-size:21px;
        font-weight:750;
        color:#0F172A;
    ">
    Upload a breast ultrasound
    </div>

    <div style="
        color:#64748B;
        margin-top:8px;
    ">
    JPG, JPEG, PNG, BMP or TIFF
    </div>

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

    try:

        with st.spinner(
            "Analyzing ultrasound..."
        ):

            (
                prediction,
                predicted_index,
                probabilities,
                cam
            ) = predict(
                image,
                model
            )

    except Exception as e:

        st.error(
            "Grad-CAM analysis failed."
        )

        with st.expander(
            "Technical details"
        ):

            st.code(
                str(e)
            )

        st.stop()

    confidence = (
        float(
            probabilities.max()
        ) * 100
    )


    # ========================================================
    # PRIMARY RESULT
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    image_col, result_col = st.columns(
        [1.05, 0.95],
        gap="large"
    )


    # IMAGE

    with image_col:

        st.markdown(
            '<div class="card-title">'
            'Ultrasound'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )


    # RESULT

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

        st.markdown(
            "**Probability distribution**"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                float(
                    probabilities[i]
                ) * 100
            )

            st.write(
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
    # EXPLAINABILITY
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="explain-card">

        <div class="explain-title">
        Where is MammoSense looking?
        </div>

        <div class="explain-description">

        The attention map below provides a visual
        explanation of regions that contributed to
        the model's selected prediction.

        Red/yellow areas represent stronger model
        activation; blue areas represent weaker
        activation.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # CREATE HEATMAP

    heatmap, overlay = create_heatmap(
        image,
        cam
    )


    # ========================================================
    # GRAD-CAM VISUALS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    cam1, cam2, cam3 = st.columns(
        3,
        gap="medium"
    )


    with cam1:

        st.markdown(
            "**Original**"
        )

        st.image(
            image,
            use_container_width=True
        )


    with cam2:

        st.markdown(
            "**AI attention map**"
        )

        st.image(
            heatmap,
            use_container_width=True
        )


    with cam3:

        st.markdown(
            "**Attention overlay**"
        )

        st.image(
            overlay,
            use_container_width=True
        )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.info(
        "The heatmap shows where the model's internal "
        "features were most strongly activated for "
        "the selected class. It is an AI explainability "
        "visualization, not a clinical lesion segmentation."
    )


    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        'Analysis overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        3
    )


    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "**Top prediction**"
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


    with col2:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "**Model probability**"
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


    with col3:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "**Explainability**"
        )

        st.markdown(
            "### ViT Grad-CAM"
        )

        st.caption(
            "Attention visualization enabled"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

with st.expander(
    "MammoSense model information"
):

    c1, c2, c3 = st.columns(
        3
    )

    with c1:

        st.markdown(
            "**Architecture**"
        )

        st.write(
            "ViT-Small Patch16-224"
        )

    with c2:

        st.markdown(
            "**Dataset**"
        )

        st.write(
            "BUSI"
        )

    with c3:

        st.markdown(
            "**Classes**"
        )

        st.write(
            "Normal / Benign / Malignant"
        )

    st.divider()

    st.write(
        f"**Model file:** `{model_filename}`"
    )

    st.write(
        "**Input resolution:** 224 × 224"
    )

    st.write(
        "**Framework:** PyTorch + timm"
    )

    st.write(
        "**Explainability:** ViT Grad-CAM"
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer">

<strong>⚠️ Research & Medical Disclaimer</strong>

<br><br>

MammoSense is an experimental AI research prototype
for breast ultrasound image classification.

It is <strong>not a medical device</strong> and does not
provide a medical diagnosis.

A prediction of <strong>Malignant</strong> does not by
itself establish that a patient has breast cancer.
Similarly, a <strong>Normal</strong> or
<strong>Benign</strong> prediction does not definitively
exclude disease.

The Grad-CAM visualization indicates regions associated
with the model's prediction. It should not be interpreted
as a definitive lesion boundary or clinical finding.

All clinical decisions should be made by qualified
healthcare professionals using appropriate medical
evaluation and diagnostic procedures.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

MammoSense · Explainable Breast Ultrasound AI

<br>

ViT-Small Patch16-224 · BUSI · PyTorch

</div>
""", unsafe_allow_html=True)
