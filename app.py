# ============================================================
# MAMMOSENSE
# AI-ASSISTED BREAST ULTRASOUND ANALYSIS
# ViT-Small Patch16-224
# ============================================================

import io
import os
import numpy as np
import cv2
import streamlit as st

import torch
import torch.nn as nn
import timm
import torchvision.transforms as T

from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage
)
from reportlab.lib.units import inch


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MammoSense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
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
            circle at 10% 10%,
            rgba(99,102,241,0.13),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(236,72,153,0.10),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #070a12,
            #0b1020 55%,
            #090d18
        );

    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 36px 40px;
    border-radius: 26px;
    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.96),
            rgba(15,23,42,0.92)
        );

    border: 1px solid rgba(148,163,184,0.16);

    box-shadow:
        0 25px 70px rgba(0,0,0,0.30);

    margin-bottom: 25px;
}

.brand {
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.05;
    white-space: nowrap;
}

.hero-subtitle {
    margin-top: 10px;
    font-size: 18px;
    color: #94a3b8;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 15px;
}

.card {
    padding: 24px;
    border-radius: 20px;
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(148,163,184,0.13);
    margin-bottom: 20px;
}

.result-card {
    padding: 30px;
    border-radius: 22px;
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.96),
            rgba(15,23,42,0.88)
        );

    border: 1px solid rgba(148,163,184,0.15);
    text-align: center;
}

.result-label {
    color: #94a3b8;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.6px;
}

.result-value {
    font-size: 38px;
    font-weight: 800;
    margin: 8px 0;
}

.confidence-value {
    font-size: 25px;
    font-weight: 700;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.info-box {
    padding: 20px;
    border-radius: 17px;
    background: rgba(30,41,59,0.65);
    border: 1px solid rgba(148,163,184,0.12);
}

.warning {
    padding: 20px;
    border-radius: 17px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
}

.success-box {
    padding: 20px;
    border-radius: 17px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

REPO_ID = "Makky07/MammoSense-breast-ultrasound"

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

<div class="brand">
MammoSense
</div>

<div class="hero-subtitle">
Vision Transformer-powered breast ultrasound image analysis
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ARCHITECTURE
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
# FIND MODEL AUTOMATICALLY
# ============================================================

@st.cache_resource
def find_model_file():

    api = HfApi()

    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="model"
    )

    model_files = [
        f for f in files
        if f.lower().endswith(".pt")
        or f.lower().endswith(".pth")
        or f.lower().endswith(".bin")
    ]

    if not model_files:

        raise FileNotFoundError(
            "No PyTorch model file (.pt/.pth/.bin) "
            "was found in the Hugging Face repository."
        )

    # Prefer files containing model-related names
    preferred = [
        f for f in model_files
        if any(
            word in f.lower()
            for word in [
                "model",
                "vit",
                "busi",
                "mammosense"
            ]
        )
    ]

    if preferred:

        return preferred[0]

    return model_files[0]


# ============================================================
# DOWNLOAD MODEL
# ============================================================

@st.cache_resource
def download_model():

    filename = find_model_file()

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model"
    )

    return model_path, filename


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path, filename = download_model()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Your training code saved:
    #
    # {
    #     "model_state_dict": ...,
    #     "class_names": ...,
    #     "class_to_idx": ...,
    #     ...
    # }
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        num_classes = checkpoint.get(
            "num_classes",
            3
        )

    else:

        # Also support a raw state_dict
        state_dict = checkpoint
        num_classes = 3

    model = BUSIViT(
        num_classes=num_classes
    )

    # --------------------------------------------------------
    # Clean prefixes
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "model."
        ):

            new_key = new_key[
                len("model.") :
            ]

        cleaned_state_dict[
            new_key
        ] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        checkpoint,
        filename
    )


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = T.Compose([

    T.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
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
# GRAD-CAM
# ============================================================

def generate_vit_gradcam(
    model,
    image_tensor,
    class_index
):

    activations = []
    gradients = []

    target_layer = (
        model.backbone.blocks[-1]
    )

    def forward_hook(
        module,
        inputs,
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

        if grad_output:

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
                "No Grad-CAM activations captured."
            )

        if not gradients:

            raise RuntimeError(
                "No Grad-CAM gradients captured."
            )

        activation = activations[0]
        gradient = gradients[0]

        # ----------------------------------------------------
        # Expected:
        # [Batch, Tokens, Embedding]
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
            grid_size * grid_size
            != num_patches
        ):

            raise RuntimeError(
                f"Cannot form Grad-CAM grid "
                f"from {num_patches} patches."
            )

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        weights = gradient.mean(
            dim=1,
            keepdim=True
        )

        cam = (
            activation * weights
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
            cam.detach()
            .float()
            .cpu()
            .numpy()
        )

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:

            cam /= maximum

        return cam

    finally:

        forward_handle.remove()
        backward_handle.remove()

        model.zero_grad(
            set_to_none=True
        )


# ============================================================
# CREATE GRAD-CAM OVERLAY
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
        (
            width,
            height
        ),
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
# PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    predicted_label,
    confidence,
    probabilities,
    heatmap=None,
    overlay=None
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14
    )

    story = []

    story.append(
        Paragraph(
            "MammoSense",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted Breast Ultrasound Analysis",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "ANALYSIS SUMMARY",
            heading_style
        )
    )

    result_data = [
        ["Prediction", predicted_label],
        [
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        ],
        [
            "Model",
            "ViT-Small Patch16-224"
        ],
        [
            "Dataset",
            "BUSI Breast Ultrasound Images"
        ]
    ]

    result_table = Table(
        result_data,
        colWidths=[
            160,
            300
        ]
    )

    result_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#EEF2FF"
                )
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#D1D5DB"
                )
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                "Helvetica"
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(
        result_table
    )

    story.append(
        Paragraph(
            "PREDICTION PROBABILITIES",
            heading_style
        )
    )

    probability_data = [
        [
            "Class",
            "Probability"
        ]
    ]

    for i, name in enumerate(
        CLASS_NAMES
    ):

        probability_data.append([
            name,
            f"{probabilities[i] * 100:.2f}%"
        ])

    probability_table = Table(
        probability_data,
        colWidths=[
            250,
            210
        ]
    )

    probability_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#111827"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor(
                    "#D1D5DB"
                )
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(
        probability_table
    )

    story.append(
        Paragraph(
            "UPLOADED ULTRASOUND",
            heading_style
        )
    )

    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)

    story.append(
        PDFImage(
            image_buffer,
            width=4.8 * inch,
            height=4.8 * inch
        )
    )

    if overlay is not None:

        story.append(
            Paragraph(
                "GRAD-CAM EXPLAINABILITY",
                heading_style
            )
        )

        overlay_buffer = io.BytesIO()

        Image.fromarray(
            overlay
        ).save(
            overlay_buffer,
            format="PNG"
        )

        overlay_buffer.seek(0)

        story.append(
            PDFImage(
                overlay_buffer,
                width=4.8 * inch,
                height=4.8 * inch
            )
        )

        story.append(
            Spacer(
                1,
                8
            )
        )

        story.append(
            Paragraph(
                "The Grad-CAM visualization highlights "
                "image regions that contributed to the "
                "model's selected prediction. It should "
                "not be interpreted as a clinical lesion "
                "segmentation.",
                normal_style
            )
        )

    story.append(
        Paragraph(
            "IMPORTANT MEDICAL DISCLAIMER",
            heading_style
        )
    )

    story.append(
        Paragraph(
            """
            MammoSense is an AI research and educational
            prototype and is not a medical diagnostic device.
            Its predictions should not be used to diagnose,
            exclude, or treat breast cancer. A qualified
            radiologist or other appropriate healthcare
            professional must interpret ultrasound images
            and make clinical decisions. Model performance
            on external clinical images may differ from its
            performance on the training dataset.
            """,
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            "MammoSense • AI-assisted breast ultrasound research",
            subtitle_style
        )
    )

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🩺 MammoSense"
    )

    st.markdown("---")

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
        "Normal • Benign • Malignant"
    )

    st.markdown(
        "### Input"
    )

    st.write(
        "Breast ultrasound"
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

    st.markdown("---")

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

        model, checkpoint, model_filename = (
            load_model()
        )

except Exception as e:

    st.error(
        "Unable to load the MammoSense model."
    )

    st.error(
        "Check that your Hugging Face repository "
        "contains the trained .pt checkpoint."
    )

    with st.expander(
        "Technical error"
    ):

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

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        "**MODEL STATUS**"
    )

    st.success(
        "● ONLINE"
    )

with col2:

    st.markdown(
        "**ARCHITECTURE**"
    )

    st.write(
        "ViT-Small"
    )

with col3:

    st.markdown(
        "**CLASSES**"
    )

    st.write(
        "3-class"
    )

with col4:

    st.markdown(
        "**CHECKPOINT**"
    )

    st.write(
        "Loaded automatically"
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Upload Ultrasound Image'
    '</div>',
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
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown(
        '<div class="section-title">'
        'Image Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    image_col, result_col = st.columns(
        [1.15, 1]
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded ultrasound image",
            use_container_width=True
        )

    image_tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        model.eval()

        with torch.no_grad():

            logits = model(
                image_tensor
            )

            probabilities_tensor = (
                torch.softmax(
                    logits,
                    dim=1
                )
            )

            predicted_class = (
                probabilities_tensor
                .argmax(
                    dim=1
                )
                .item()
            )

        probabilities = (
            probabilities_tensor[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        predicted_label = (
            CLASS_NAMES[
                predicted_class
            ]
        )

        confidence = float(
            probabilities[
                predicted_class
            ]
        )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(e)
        )

        st.stop()

    # ========================================================
    # RESULT
    # ========================================================

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
            confidence
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

        probability = float(
            probabilities[i]
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
                probability
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

    heatmap = None
    overlay = None

    if show_gradcam:

        st.markdown(
            '<div class="section-title">'
            '🔍 AI Explainability'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Grad-CAM highlights regions that "
            "contributed to the selected model prediction."
        )

        try:

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

            cam_col1, cam_col2 = st.columns(2)

            with cam_col1:

                st.image(
                    heatmap,
                    caption="MammoSense attention map",
                    use_container_width=True
                )

            with cam_col2:

                st.image(
                    overlay,
                    caption="AI attention overlay",
                    use_container_width=True
                )

        except Exception as e:

            st.warning(
                "Prediction worked, but Grad-CAM "
                "could not be generated."
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
            "The model classified this image as "
            "Normal. This means the model did not "
            "identify a pattern strongly associated "
            "with the Benign or Malignant classes."
        )

    elif predicted_label == "Benign":

        st.info(
            "The model classified this image as "
            "Benign. Benign findings are non-cancerous "
            "abnormalities, although clinical assessment "
            "may still be required."
        )

    else:

        st.warning(
            "The model classified this image as "
            "Malignant. This can be associated with "
            "cancer, but the AI result is NOT a diagnosis "
            "and requires professional clinical assessment."
        )

    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📄 Analysis Report'
        '</div>',
        unsafe_allow_html=True
    )

    try:

        pdf_bytes = create_pdf_report(
            image=image,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            heatmap=heatmap,
            overlay=overlay
        )

        st.download_button(
            label="⬇️ Download Standard PDF Report",
            data=pdf_bytes,
            file_name="MammoSense_Analysis_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            "Could not generate the PDF report."
        )

        with st.expander(
            "Report technical details"
        ):

            st.code(
                str(e)
            )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    '<div class="warning">',
    unsafe_allow_html=True
