# ============================================================
# MAMMOSENSE
# AI-ASSISTED BREAST ULTRASOUND ANALYSIS
# ViT-Small Patch16-224
# ============================================================

import io
import os
import glob

import cv2
import numpy as np
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
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MammoSense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "Makky07/MammoSense-breast-ultrasound"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99,102,241,0.14),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(236,72,153,0.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #070a12 0%,
            #0b1020 50%,
            #090d18 100%
        );
    color: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.brand {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -2px;
    white-space: nowrap;
}

.brand span {
    color: #a5b4fc;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 4px;
}

.hero {
    padding: 30px;
    border-radius: 25px;
    margin-bottom: 25px;
    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.96),
            rgba(15,23,42,0.90)
        );
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 25px 70px rgba(0,0,0,0.30);
}

.badge {
    display: inline-block;
    padding: 6px 13px;
    border-radius: 999px;
    background: rgba(99,102,241,0.16);
    color: #c7d2fe;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
}

.card {
    padding: 24px;
    border-radius: 20px;
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.12);
    margin-bottom: 20px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin: 25px 0 15px 0;
}

.result-card {
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    background:
        linear-gradient(
            145deg,
            rgba(30,41,59,0.98),
            rgba(15,23,42,0.90)
        );
    border: 1px solid rgba(148,163,184,0.14);
}

.result-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.result-value {
    font-size: 38px;
    font-weight: 800;
    margin: 8px 0 15px 0;
}

.confidence {
    font-size: 24px;
    font-weight: 700;
}

.info-box {
    padding: 18px;
    border-radius: 17px;
    background: rgba(30,41,59,0.60);
    border: 1px solid rgba(148,163,184,0.12);
}

.warning {
    padding: 20px;
    border-radius: 18px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 35px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="badge">
AI-ASSISTED BREAST ULTRASOUND ANALYSIS
</div>

<div class="brand">
Mammo<span>Sense</span>
</div>

<div class="subtitle">
Vision Transformer-powered analysis of breast ultrasound images
</div>

</div>
""",
    unsafe_allow_html=True,
)


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
            num_classes=0,
        )

        self.head = nn.Sequential(
            nn.Linear(
                self.backbone.embed_dim,
                1024,
            ),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(
                1024,
                512,
            ),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(
                512,
                num_classes,
            ),
        )

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# AUTOMATICALLY FIND .PT MODEL ON HUGGING FACE
# ============================================================

@st.cache_data
def find_model_filename():

    api = HfApi()

    files = api.list_repo_files(
        repo_id=REPO_ID,
        repo_type="model",
    )

    pt_files = [
        f for f in files
        if f.lower().endswith(".pt")
    ]

    if not pt_files:

        raise FileNotFoundError(
            "No .pt model checkpoint was found "
            "in the Hugging Face repository."
        )

    # Prefer the known checkpoint if it exists,
    # otherwise use the first .pt file.
    preferred = [
        f for f in pt_files
        if "gaia" in f.lower()
        or "busi" in f.lower()
        or "vit" in f.lower()
        or "model" in f.lower()
    ]

    if preferred:
        return preferred[0]

    return pt_files[0]


# ============================================================
# DOWNLOAD MODEL
# ============================================================

@st.cache_resource
def download_model():

    filename = find_model_filename()

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="model",
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
    )

    if isinstance(checkpoint, dict):

        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if state_dict is None:

            state_dict = checkpoint.get(
                "state_dict"
            )

        if state_dict is None:

            # Some checkpoints may themselves
            # contain the state dictionary.
            state_dict = checkpoint

        num_classes = checkpoint.get(
            "num_classes",
            3,
        )

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    model = BUSIViT(
        num_classes=num_classes,
    )

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        if new_key.startswith("model."):
            new_key = new_key[6:]

        cleaned[new_key] = value

    try:

        model.load_state_dict(
            cleaned,
            strict=True,
        )

    except RuntimeError as e:

        raise RuntimeError(
            "The checkpoint was found, but its "
            "weights do not match the MammoSense "
            "ViT-Small architecture.\n\n"
            + str(e)
        )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint, filename


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = T.Compose(
    [
        T.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        T.ToTensor(),

        T.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# VIT GRAD-CAM
# ============================================================

def generate_vit_gradcam(
    model,
    image_tensor,
    class_index,
):

    activations = []
    gradients = []

    target_layer = model.backbone.blocks[-1]

    def forward_hook(
        module,
        inputs,
        output,
    ):

        activations.append(output)

    def backward_hook(
        module,
        grad_input,
        grad_output,
    ):

        gradients.append(
            grad_output[0]
        )

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
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
                "No ViT activations captured."
            )

        if not gradients:

            raise RuntimeError(
                "No ViT gradients captured."
            )

        activation = activations[0]
        gradient = gradients[0]

        if activation.ndim != 3:

            raise RuntimeError(
                "Unexpected activation shape: "
                + str(
                    tuple(
                        activation.shape
                    )
                )
            )

        # Remove CLS token
        activation = activation[:, 1:, :]
        gradient = gradient[:, 1:, :]

        num_patches = activation.shape[1]

        grid_size = int(
            np.sqrt(num_patches)
        )

        if grid_size * grid_size != num_patches:

            raise RuntimeError(
                f"Invalid ViT patch grid: "
                f"{num_patches}"
            )

        weights = gradient.mean(
            dim=1,
            keepdim=True,
        )

        cam = (
            activation * weights
        ).sum(dim=2)

        cam = torch.relu(cam)

        cam = cam.reshape(
            grid_size,
            grid_size,
        )

        cam = (
            cam.detach()
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
    cam,
):

    original = np.array(
        image.convert("RGB")
    )

    height, width = original.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0,
    )

    return heatmap, overlay


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    filename,
    prediction,
    confidence,
    probabilities,
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=25,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
    )

    story = []

    story.append(
        Paragraph(
            "MammoSense",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted Breast Ultrasound Analysis Report",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "Analysis Summary",
            heading_style,
        )
    )

    summary_data = [
        ["Image", filename],
        ["AI Classification", prediction],
        [
            "Model Confidence",
            f"{confidence * 100:.2f}%",
        ],
        [
            "Model",
            "ViT-Small Patch16-224",
        ],
        [
            "Dataset",
            "BUSI Breast Ultrasound Dataset",
        ],
    ]

    table = Table(
        summary_data,
        colWidths=[150, 320],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#EEF2FF"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#172033"),
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(table)

    story.append(
        Paragraph(
            "Prediction Probabilities",
            heading_style,
        )
    )

    probability_data = [
        ["Classification", "Probability"]
    ]

    for i, name in enumerate(CLASS_NAMES):

        probability_data.append(
            [
                name,
                f"{probabilities[i] * 100:.2f}%",
            ]
        )

    prob_table = Table(
        probability_data,
        colWidths=[250, 220],
    )

    prob_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#E0E7FF"),
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D1D5DB"),
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(prob_table)

    story.append(
        Paragraph(
            "Medical Disclaimer",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "MammoSense is an AI research and "
            "educational prototype and is not a "
            "medical diagnostic device. Its output "
            "must not be used as a standalone diagnosis "
            "or to determine treatment. Breast ultrasound "
            "images should be interpreted by a qualified "
            "healthcare professional. Model performance "
            "may differ on images outside the training "
            "dataset.",
            body_style,
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Generated by MammoSense",
            subtitle_style,
        )
    )

    document.build(story)

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
        "### Classification"
    )

    st.write(
        "Normal\n\n"
        "Benign\n\n"
        "Malignant"
    )

    st.markdown(
        "### Input"
    )

    st.write(
        "Breast ultrasound image"
    )

    st.markdown(
        "### Explainability"
    )

    show_gradcam = st.checkbox(
        "Generate Grad-CAM",
        value=True,
    )

    st.markdown("---")

    if DEVICE.type == "cuda":

        st.success(
            "GPU acceleration enabled"
        )

    else:

        st.info(
            "Running on CPU"
        )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Loading MammoSense model..."
    ):

        model, checkpoint, model_filename = (
            load_model()
        )

except Exception as error:

    st.error(
        "Unable to load the MammoSense model."
    )

    st.markdown(
        """
The application found an issue while loading
the model from the Hugging Face repository.
"""
    )

    with st.expander(
        "Technical details"
    ):

        st.code(
            str(error)
        )

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

status1, status2, status3 = st.columns(3)

with status1:

    st.markdown(
        """
<div class="info-box">
<b>MODEL STATUS</b><br>
🟢 Online
</div>
""",
        unsafe_allow_html=True,
    )

with status2:

    st.markdown(
        """
<div class="info-box">
<b>ARCHITECTURE</b><br>
ViT-Small Patch16-224
</div>
""",
        unsafe_allow_html=True,
    )

with status3:

    st.markdown(
        """
<div class="info-box">
<b>CLASSIFICATION</b><br>
3 Classes
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Upload Ultrasound Image'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a breast ultrasound image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
    ],
    label_visibility="collapsed",
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
        'Analysis'
        '</div>',
        unsafe_allow_html=True,
    )

    image_col, result_col = st.columns(
        [1.15, 1]
    )

    with image_col:

        st.image(
            image,
            caption="Uploaded ultrasound image",
            use_container_width=True,
        )

    image_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        model.eval()

        with torch.no_grad():

            logits = model(
                image_tensor
            )

            probabilities_tensor = torch.softmax(
                logits,
                dim=1,
            )

            predicted_class = (
                probabilities_tensor
                .argmax(dim=1)
                .item()
            )

        probabilities = (
            probabilities_tensor[0]
            .detach()
            .cpu()
            .numpy()
        )

        predicted_label = CLASS_NAMES[
            predicted_class
        ]

        confidence = float(
            probabilities[
                predicted_class
            ]
        )

    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.code(
            str(error)
        )

        st.stop()

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result_col:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="result-label">'
            'AI PREDICTION'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="result-value">'
            f'{predicted_label}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="result-label">'
            'MODEL CONFIDENCE'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="confidence">'
            f'{confidence * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.progress(
            confidence
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Prediction Probabilities'
        '</div>',
        unsafe_allow_html=True,
    )

    probability_columns = st.columns(3)

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = float(
            probabilities[index]
        )

        with probability_columns[index]:

            st.markdown(
                '<div class="info-box">',
                unsafe_allow_html=True,
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
                unsafe_allow_html=True,
            )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    if show_gradcam:

        st.markdown(
            '<div class="section-title">'
            '🔍 AI Explainability'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Grad-CAM provides a visual indication "
            "of regions that contributed to the "
            "selected model prediction. It is "
            "not a diagnostic map."
        )

        try:

            cam = generate_vit_gradcam(
                model,
                image_tensor,
                predicted_class,
            )

            heatmap, overlay = (
                create_gradcam_overlay(
                    image,
                    cam,
                )
            )

            cam1, cam2 = st.columns(2)

            with cam1:

                st.image(
                    heatmap,
                    caption="MammoSense attention map",
                    use_container_width=True,
                )

            with cam2:

                st.image(
                    overlay,
                    caption="AI attention overlay",
                    use_container_width=True,
                )

        except Exception as error:

            st.warning(
                "Prediction completed, but Grad-CAM "
                "could not be generated."
            )

            with st.expander(
                "Grad-CAM technical details"
            ):

                st.code(
                    str(error)
                )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Understanding the Classification'
        '</div>',
        unsafe_allow_html=True,
    )

    if predicted_label == "Normal":

        st.info(
            "The model classified the uploaded "
            "image as Normal. This means the model "
            "did not identify a pattern strongly "
            "associated with the Benign or Malignant "
            "classes."
        )

    elif predicted_label == "Benign":

        st.info(
            "The model classified the uploaded image "
            "as Benign. Benign findings are "
            "non-cancerous abnormalities, although "
            "clinical assessment may still be required."
        )

    else:

        st.warning(
            "The model classified the uploaded image "
            "as Malignant. This result does NOT "
            "constitute a cancer diagnosis. A qualified "
            "healthcare professional must evaluate the "
            "ultrasound and determine the appropriate "
            "next step."
        )


    # ========================================================
    # PDF REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Report'
        '</div>',
        unsafe_allow_html=True,
    )

    report_bytes = create_pdf_report(
        filename=uploaded_file.name,
        prediction=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
    )

    st.download_button(
        label="📄 Download Standard PDF Report",
        data=report_bytes,
        file_name="MammoSense_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    '<div class="warning">',
    unsafe_allow_html=True,
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

Images from patients or sources outside the training
dataset may differ substantially from the training
data, so real-world performance may be different.
"""
)

st.markdown(
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "ℹ️ About MammoSense"
):

    st.write(
        "**Architecture:** ViT-Small Patch16-224"
    )

    st.write(
        "**Input:** Breast ultrasound image"
    )

    st.write(
        "**Input resolution:** 224 × 224"
    )

    st.write(
        "**Classes:** Normal, Benign, Malignant"
    )

    st.write(
        "**Training dataset:** BUSI"
    )

    st.write(
        "**Explainability:** ViT Grad-CAM"
    )

    st.write(
        f"**Hugging Face checkpoint:** "
        f"{model_filename}"
    )

    st.write(
        f"**Runtime:** {DEVICE}"
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
    unsafe_allow_html=True,
)
