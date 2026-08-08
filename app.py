import os
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from huggingface_hub import hf_hub_download

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MammoSense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ID = "Makky07/MammoSense-breast-ultrasound"

# IMPORTANT:
# This is the exact filename visible in your Hugging Face repository.
MODEL_FILENAME = "gaia_busi_vit_small (1).pt"

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------------------------------------------------------
   APP BACKGROUND
--------------------------------------------------------- */

.stApp {
    background:
        linear-gradient(
            90deg,
            rgba(7, 16, 42, 0.97) 0%,
            rgba(7, 16, 42, 0.93) 42%,
            rgba(7, 16, 42, 0.72) 100%
        ),
        url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=2000&q=85");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ---------------------------------------------------------
   MAIN CONTENT
--------------------------------------------------------- */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------------------------------------------------------
   HEADER
--------------------------------------------------------- */

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 8px;
}

.brand-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    box-shadow: 0 10px 30px rgba(236,72,153,0.25);
}

.brand-name {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: white;
    white-space: nowrap;
}

.brand-tagline {
    color: rgba(255,255,255,0.68);
    font-size: 15px;
    margin-top: 2px;
}

/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero {
    margin-top: 30px;
    padding: 48px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            rgba(20, 30, 75, 0.92),
            rgba(61, 31, 100, 0.80)
        );
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 25px 70px rgba(0,0,0,0.30);
}

.hero h1 {
    color: white;
    font-size: 45px;
    line-height: 1.05;
    margin-bottom: 15px;
    letter-spacing: -1.8px;
}

.hero p {
    color: rgba(255,255,255,0.75);
    font-size: 18px;
    max-width: 720px;
    line-height: 1.7;
}

/* ---------------------------------------------------------
   CARDS
--------------------------------------------------------- */

.card {
    background: rgba(255,255,255,0.96);
    border-radius: 22px;
    padding: 26px;
    margin-top: 22px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.20);
}

.card-title {
    font-size: 22px;
    font-weight: 750;
    color: #172033;
    margin-bottom: 7px;
}

.card-subtitle {
    font-size: 14px;
    color: #667085;
    margin-bottom: 20px;
}

/* ---------------------------------------------------------
   RESULT
--------------------------------------------------------- */

.result-card {
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.98),
        rgba(248,250,252,0.98)
    );
    border-radius: 24px;
    padding: 30px;
    margin-top: 22px;
    box-shadow: 0 18px 50px rgba(0,0,0,0.20);
}

.result-label {
    color: #667085;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 700;
}

.result-value {
    font-size: 40px;
    font-weight: 800;
    margin-top: 5px;
    color: #111827;
}

.confidence-text {
    color: #667085;
    font-size: 15px;
}

/* ---------------------------------------------------------
   DISCLAIMER
--------------------------------------------------------- */

.disclaimer {
    background: rgba(255,248,235,0.98);
    border-left: 5px solid #f59e0b;
    padding: 18px 20px;
    border-radius: 14px;
    margin-top: 22px;
    color: #573b08;
    line-height: 1.6;
    font-size: 14px;
}

/* ---------------------------------------------------------
   INFO
--------------------------------------------------------- */

.info-box {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 20px;
    color: white;
}

.info-box h4 {
    margin-top: 0;
}

.info-box p {
    color: rgba(255,255,255,0.72);
    line-height: 1.6;
}

/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {
    text-align: center;
    color: rgba(255,255,255,0.55);
    padding: 40px 0 10px;
    font-size: 13px;
}

/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 48px;
}

/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 700px) {

    .brand-name {
        font-size: 30px;
    }

    .brand-icon {
        width: 44px;
        height: 44px;
        font-size: 22px;
    }

    .hero {
        padding: 28px;
    }

    .hero h1 {
        font-size: 32px;
    }

    .hero p {
        font-size: 15px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="brand">
    <div class="brand-icon">🩺</div>
    <div>
        <div class="brand-name">MammoSense</div>
        <div class="brand-tagline">
            AI-assisted breast ultrasound analysis
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<h1>
Intelligent breast ultrasound analysis.
</h1>

<p>
MammoSense uses a Vision Transformer trained on the
Breast Ultrasound Images (BUSI) dataset to classify
ultrasound images into Normal, Benign, or Malignant
categories and provide visual interpretability through
Grad-CAM.
</p>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# MODEL ARCHITECTURE
# EXACTLY MATCHES TRAINING
# ============================================================

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            self.alpha
            * (1 - pt) ** self.gamma
            * ce_loss
        )

        return focal_loss.mean()


class BUSIViT(nn.Module):

    def __init__(
        self,
        num_classes=3
    ):

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

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                1024,
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        features = self.backbone(x)

        return self.head(features)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

@st.cache_resource
def download_model():

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = download_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
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

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    return model, checkpoint


# ============================================================
# TRANSFORM
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

def generate_gradcam(
    model,
    image_tensor,
    target_class
):

    activations = []
    gradients = []

    # Last transformer block
    target_layer = (
        model
        .backbone
        .blocks[-1]
        .norm1
    )

    def forward_hook(module, inp, output):

        activations.append(output)

        if output.requires_grad:

            output.register_hook(
                lambda grad:
                gradients.append(grad)
            )

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    model.zero_grad()

    logits = model(
        image_tensor
    )

    score = logits[
        0,
        target_class
    ]

    score.backward()

    handle.remove()

    if (
        len(activations) == 0
        or len(gradients) == 0
    ):
        return None

    activation = activations[0].detach()

    gradient = gradients[0].detach()

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

    # Global average pooling
    weights = gradient.mean(
        dim=1,
        keepdim=True
    )

    cam = (
        activation * weights
    ).sum(
        dim=-1
    )

    cam = F.relu(cam)

    # 196 patches = 14 x 14
    num_patches = cam.shape[-1]

    grid_size = int(
        np.sqrt(num_patches)
    )

    cam = cam.reshape(
        1,
        grid_size,
        grid_size
    )

    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze()

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    return cam.cpu().numpy()


# ============================================================
# CREATE VISUAL GRAD-CAM
# ============================================================

def create_gradcam_image(
    original_image,
    heatmap
):

    image = np.array(
        original_image.convert("RGB")
    )

    h, w = image.shape[:2]

    heatmap_img = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    ).resize(
        (w, h)
    )

    heatmap_array = np.array(
        heatmap_img
    )

    # Simple red/yellow activation map
    import cv2

    colored = cv2.applyColorMap(
        heatmap_array,
        cv2.COLORMAP_JET
    )

    colored = cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB
    )

    overlay = (
        0.55 * image
        +
        0.45 * colored
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        overlay
    )


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf(
    image,
    prediction,
    probabilities,
    heatmap_image
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
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        leading=30,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=15,
        spaceBefore=14,
        spaceAfter=8
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
            "Breast Ultrasound AI Analysis Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "<b>AI Classification Result</b>",
            heading_style
        )
    )

    result_table = Table(
        [
            ["Prediction", prediction],
            [
                "Normal",
                f"{probabilities[0] * 100:.2f}%"
            ],
            [
                "Benign",
                f"{probabilities[1] * 100:.2f}%"
            ],
            [
                "Malignant",
                f"{probabilities[2] * 100:.2f}%"
            ]
        ],
        colWidths=[
            180,
            260
        ]
    )

    result_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172554")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
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
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.lightgrey
            ),

            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.whitesmoke
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                9
            )
        ])
    )

    story.append(
        result_table
    )

    story.append(
        Paragraph(
            "Input Ultrasound",
            heading_style
        )
    )

    img_buffer = io.BytesIO()

    image.save(
        img_buffer,
        format="PNG"
    )

    img_buffer.seek(0)

    story.append(
        RLImage(
            img_buffer,
            width=4.8 * inch,
            height=4.8 * inch
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    if heatmap_image is not None:

        story.append(
            Paragraph(
                "Grad-CAM Interpretability",
                heading_style
            )
        )

        cam_buffer = io.BytesIO()

        heatmap_image.save(
            cam_buffer,
            format="PNG"
        )

        cam_buffer.seek(0)

        story.append(
            RLImage(
                cam_buffer,
                width=4.8 * inch,
                height=4.8 * inch
            )
        )

    story.append(
        Paragraph(
            "Model Information",
            heading_style
        )
    )

    story.append(
        Paragraph(
            """
            Architecture: Vision Transformer Small
            (ViT-Small Patch16-224).<br/>
            Dataset: Breast Ultrasound Images (BUSI).<br/>
            Classes: Normal, Benign, Malignant.<br/>
            Input resolution: 224 × 224 pixels.
            """,
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "Important Medical Disclaimer",
            heading_style
        )
    )

    story.append(
        Paragraph(
            """
            MammoSense is an experimental AI research tool and
            is not a medical device or a substitute for a qualified
            radiologist, physician, biopsy, or other clinical
            assessment. AI predictions may be incorrect.
            This report should not be used alone to make medical
            decisions.
            """,
            styles["Normal"]
        )
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    story.append(
        Paragraph(
            "Generated by MammoSense",
            subtitle_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Initializing MammoSense AI model..."
    ):

        model, checkpoint = load_model()

    st.success(
        "MammoSense AI model loaded successfully."
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
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
<div class="card">

<div class="card-title">
Upload ultrasound image
</div>

<div class="card-subtitle">
Upload a breast ultrasound image in JPG, JPEG or PNG format.
</div>

</div>
""",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose ultrasound image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )

    with col1:

        st.markdown(
            """
            <div class="card">
            <div class="card-title">
            Ultrasound Image
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0).to(
        DEVICE
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing ultrasound..."
    ):

        model.eval()

        with torch.no_grad():

            logits = model(
                input_tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

            predicted_class = int(
                probabilities.argmax().item()
            )

        # Grad-CAM requires gradients
        heatmap = generate_gradcam(
            model,
            input_tensor,
            predicted_class
        )

    prediction = CLASS_NAMES[
        predicted_class
    ]

    probs = probabilities.detach().cpu().numpy()

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    cam_image = None

    if heatmap is not None:

        cam_image = create_gradcam_image(
            image,
            heatmap
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with col2:

        st.markdown(
            f"""
            <div class="result-card">

            <div class="result-label">
            AI classification
            </div>

            <div class="result-value">
            {prediction}
            </div>

            <div class="confidence-text">
            Model confidence:
            <b>{probs[predicted_class] * 100:.2f}%</b>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # PROBABILITY BARS
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="card">
            <div class="card-title">
            Probability distribution
            </div>
            """,
            unsafe_allow_html=True
        )

        for name, probability in zip(
            CLASS_NAMES,
            probs
        ):

            st.write(
                f"**{name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    if cam_image is not None:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            Model attention — Grad-CAM
            </div>

            <div class="card-subtitle">
            Highlighted regions indicate areas that contributed
            to the model's classification.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        gc1, gc2 = st.columns(
            2,
            gap="large"
        )

        with gc1:

            st.image(
                image,
                caption="Original ultrasound",
                use_container_width=True
            )

        with gc2:

            st.image(
                cam_image,
                caption="MammoSense Grad-CAM",
                use_container_width=True
            )

    # ========================================================
    # MEDICAL INFORMATION
    # ========================================================

    st.markdown(
        """
        <div class="info-box">

        <h4>Understanding the result</h4>

        <p>
        <b>Normal:</b> The model did not identify features that
        strongly correspond to the abnormal patterns represented
        in its training data.
        </p>

        <p>
        <b>Benign:</b> A benign lesion is non-cancerous.
        Benign breast conditions can still require clinical
        assessment and follow-up.
        </p>

        <p>
        <b>Malignant:</b> A malignant lesion is cancerous and
        requires clinical evaluation and confirmation.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="disclaimer">

        <b>Important:</b> MammoSense is an experimental AI
        research tool. It does not diagnose breast cancer and
        must not replace evaluation by a qualified radiologist
        or physician. A prediction from this application should
        never be used by itself to make a medical decision.

        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # REPORT
    # ========================================================

    pdf_bytes = generate_pdf(
        image,
        prediction,
        probs,
        cam_image
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.download_button(
        label="📄 Download Standard Analysis Report",
        data=pdf_bytes,
        file_name="MammoSense_Analysis_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "MammoSense model information"
):

    st.write(
        """
        **Architecture:** ViT-Small Patch16-224

        **Training dataset:** BUSI Breast Ultrasound Images

        **Classes:** Normal / Benign / Malignant

        **Input size:** 224 × 224

        **Interpretability:** Vision Transformer Grad-CAM

        **Framework:** PyTorch + timm

        **Model repository:** Makky07/MammoSense-breast-ultrasound
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

MammoSense • AI-assisted breast ultrasound research

<br><br>

For research and educational use only.

</div>
""",
    unsafe_allow_html=True
)
