# ============================================================
# MAMMOSENSE
# AI-ASSISTED BREAST ULTRASOUND ANALYSIS
# ViT-Small Patch16-224
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2

from PIL import Image
from huggingface_hub import hf_hub_download
import torchvision.transforms as T


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MammoSense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(99,102,241,0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(236,72,153,0.08),
            transparent 30%
        ),
        #080b14;
    color: #f8fafc;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 35px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.95),
            rgba(15,23,42,0.90)
        );
    border: 1px solid rgba(148,163,184,0.15);
    margin-bottom: 25px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 18px;
    color: #94a3b8;
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 15px;
}

.card {
    padding: 24px;
    border-radius: 20px;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.13);
    margin-bottom: 20px;
}

.result-card {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(
        145deg,
        rgba(30,41,59,0.95),
        rgba(15,23,42,0.85)
    );
    border: 1px solid rgba(148,163,184,0.15);
    text-align: center;
}

.result-label {
    color: #94a3b8;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-value {
    font-size: 38px;
    font-weight: 800;
    margin: 8px 0;
}

.confidence-value {
    font-size: 24px;
    font-weight: 700;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.info-box {
    padding: 18px;
    border-radius: 16px;
    background: rgba(30,41,59,0.65);
    border: 1px solid rgba(148,163,184,0.12);
}

.warning {
    padding: 18px;
    border-radius: 16px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fbbf24;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 30px;
}

div[data-testid="stFileUploader"] {
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

REPO_ID = "Makky07/MammoSense-breast-ultrasound"
MODEL_FILENAME = "gaia_busi_vit_small.pt"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<div class="badge">
AI-ASSISTED BREAST ULTRASOUND ANALYSIS
</div>

<div class="hero-title">
MammoSense
</div>

<div class="hero-subtitle">
Vision Transformer-powered analysis of breast ultrasound images.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE
# IMPORTANT:
# This MUST match the architecture used during training.
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

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = download_model()

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    model = BUSIViT(
        num_classes=checkpoint.get(
            "num_classes",
            3
        )
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    # --------------------------------------------------------
    # Remove possible Lightning prefixes if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "model."
        ):
            new_key = new_key[6:]

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[7:]

        cleaned_state_dict[
            new_key
        ] = value

    # --------------------------------------------------------
    # Load exact trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    return model, checkpoint


# ============================================================
# IMAGE TRANSFORM
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
# GRAD-CAM FOR VISION TRANSFORMER
# ============================================================

def generate_vit_gradcam(
    model,
    image_tensor,
    class_index
):

    activations = []
    gradients = []

    # --------------------------------------------------------
    # Final transformer block
    # --------------------------------------------------------

    target_layer = (
        model.backbone.blocks[-1]
    )

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output
        )

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0]
        )

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        model.zero_grad(
            set_to_none=True
        )

        # IMPORTANT:
        # Do NOT use torch.no_grad()
        logits = model(
            image_tensor
        )

        score = logits[
            0,
            class_index
        ]

        score.backward()

        if not activations:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        if not gradients:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )

        activation = activations[0]

        gradient = gradients[0]

        # ----------------------------------------------------
        # Some timm ViT versions return:
        #
        # [B, Tokens, Embedding]
        #
        # Others can return different arrangements.
        # ----------------------------------------------------

        if activation.ndim != 3:

            raise RuntimeError(
                "Unexpected ViT activation shape: "
                + str(
                    tuple(
                        activation.shape
                    )
                )
            )

        # Remove CLS token
        activation = activation[
            :,
            1:,
            :
        ]

        gradient = gradient[
            :,
            1:,
            :
        ]

        num_patches = (
            activation.shape[1]
        )

        grid_size = int(
            np.sqrt(
                num_patches
            )
        )

        if (
            grid_size *
            grid_size
            != num_patches
        ):

            raise RuntimeError(
                f"Cannot create Grad-CAM "
                f"grid from {num_patches} "
                f"patches."
            )

        # ----------------------------------------------------
        # Grad-CAM weighting
        # ----------------------------------------------------

        weights = gradient.mean(
            dim=1,
            keepdim=True
        )

        cam = (
            activation *
            weights
        ).sum(
            dim=2
        )

        cam = torch.relu(
            cam
        )

        cam = cam.reshape(
            grid_size,
            grid_size
        )

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )

        # Normalize

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:

            cam /= max_value

        return cam

    finally:

        forward_handle.remove()

        backward_handle.remove()

        model.zero_grad(
            set_to_none=True
        )


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_gradcam_overlay(
    image,
    cam
):

    original = np.array(
        image.convert("RGB")
    )

    height, width = (
        original.shape[:2]
    )

    cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return heatmap, overlay


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🩺 MammoSense"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Model"
    )

    st.write(
        "ViT-Small Patch16-224"
    )

    st.markdown(
        "### Classes"
    )

    st.write(
        "• Normal\n"
        "• Benign\n"
        "• Malignant"
    )

    st.markdown(
        "### Input"
    )

    st.write(
        "Breast ultrasound image"
    )

    st.markdown(
        "### Image size"
    )

    st.write(
        "224 × 224 pixels"
    )

    st.markdown(
        "### Runtime"
    )

    if DEVICE.type == "cuda":

        st.success(
            "GPU acceleration"
        )

    else:

        st.info(
            "CPU inference"
        )

    st.markdown(
        "---"
    )

    show_gradcam = st.checkbox(
        "🔍 Generate Grad-CAM",
        value=True
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Loading MammoSense AI model..."
    ):

        model, checkpoint = (
            load_model()
        )

except Exception as e:

    st.error(
        "Unable to load the MammoSense model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

status_col1, status_col2, status_col3 = (
    st.columns(3)
)

with status_col1:

    st.markdown(
        "**MODEL STATUS**"
    )

    st.success(
        "● Online"
    )

with status_col2:

    st.markdown(
        "**ARCHITECTURE**"
    )

    st.write(
        "ViT-Small"
    )

with status_col3:

    st.markdown(
        "**CLASSES**"
    )

    st.write(
        "3-class classification"
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Upload Ultrasound Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a breast ultrasound image",
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


# ============================================================
# MAIN ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown(
        '<div class="section-title">Image Analysis</div>',
        unsafe_allow_html=True
    )

    image_col, result_col = (
        st.columns(
            [1.15, 1]
        )
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded ultrasound image",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREPARE IMAGE
    # --------------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        model.eval()

        # Do normal inference first
        with torch.no_grad():

            logits = model(
                image_tensor
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )
            )

            predicted_class = (
                probabilities
                .argmax(
                    dim=1
                )
                .item()
            )

        predicted_label = (
            CLASS_NAMES[
                predicted_class
            ]
        )

        confidence = (
            probabilities[
                0,
                predicted_class
            ]
            .item()
        )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(e)
        )

        st.stop()

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
            'AI PREDICTION'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="result-value">'
            f'{predicted_label}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="result-label">'
            'MODEL CONFIDENCE'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence-value">'
            f'{confidence * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(
            float(confidence)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Prediction Probabilities'
        '</div>',
        unsafe_allow_html=True
    )

    prob_cols = st.columns(3)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            probabilities[
                0,
                i
            ]
            .item()
        )

        with prob_cols[i]:

            st.markdown(
                '<div class="info-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                f"**{class_name}**"
            )

            st.progress(
                float(
                    probability
                )
            )

            st.write(
                f"{probability * 100:.2f}%"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    if show_gradcam:

        st.markdown(
            '<div class="section-title">'
            '🔍 AI Explainability'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Grad-CAM highlights image regions "
            "that contributed most strongly to "
            "the selected prediction."
        )

        try:

            # IMPORTANT:
            # Grad-CAM requires gradients.
            #
            # Therefore we deliberately DO NOT
            # use torch.no_grad() here.

            cam = generate_vit_gradcam(
                model,
                image_tensor,
                predicted_class
            )

            heatmap, overlay = (
                create_gradcam_overlay(
                    image,
                    cam
                )
            )

            cam_col1, cam_col2 = (
                st.columns(2)
            )

            with cam_col1:

                st.image(
                    heatmap,
                    caption=(
                        "MammoSense attention map"
                    ),
                    use_container_width=True
                )

            with cam_col2:

                st.image(
                    overlay,
                    caption=(
                        "AI attention overlay"
                    ),
                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                "The prediction worked, "
                "but Grad-CAM could not be "
                "generated for this model."
            )

            with st.expander(
                "Grad-CAM technical details"
            ):

                st.code(
                    str(e)
                )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Understanding the Result'
        '</div>',
        unsafe_allow_html=True
    )

    if predicted_label == "Normal":

        st.info(
            "The model classified this image "
            "as Normal. This means the model "
            "did not identify a pattern strongly "
            "associated with the benign or "
            "malignant classes."
        )

    elif predicted_label == "Benign":

        st.info(
            "The model classified this image "
            "as Benign. Benign findings are "
            "non-cancerous abnormalities. "
            "They can still require clinical "
            "assessment depending on the finding."
        )

    else:

        st.warning(
            "The model classified this image "
            "as Malignant. Malignant findings "
            "can be associated with cancer. "
            "This AI result must NOT be used "
            "as a diagnosis and requires "
            "professional clinical assessment."
        )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    '<div class="warning">',
    unsafe_allow_html=True
)

st.markdown(
    """
### ⚠️ Important Medical Disclaimer

**MammoSense is an AI research and educational
prototype, not a medical diagnostic device.**

Its predictions are generated by a machine-learning
model trained on the BUSI breast ultrasound dataset.

The output should **not** be used to diagnose,
exclude, or treat breast cancer.

A qualified radiologist or other appropriate
healthcare professional must interpret ultrasound
images and make clinical decisions.

External images may differ substantially from
the training dataset, so model performance in
real-world clinical settings may be different.
"""
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ About the MammoSense model"
):

    st.write(
        "**Architecture:** ViT-Small Patch16-224"
    )

    st.write(
        "**Input:** Breast ultrasound image"
    )

    st.write(
        "**Image size:** 224 × 224"
    )

    st.write(
        "**Classes:** Normal, Benign, Malignant"
    )

    st.write(
        "**Dataset:** BUSI Breast Ultrasound Images"
    )

    st.write(
        "**Explainability:** ViT Grad-CAM"
    )

    st.write(
        "**Model repository:** "
        "Makky07/MammoSense-breast-ultrasound"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

MammoSense • AI-assisted breast ultrasound research

For research and educational use only • Not a medical diagnosis

</div>
""",
    unsafe_allow_html=True
)
