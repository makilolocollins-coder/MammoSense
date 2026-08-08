# ============================================================
# MAMMOSENSE
# AI-ASSISTED BREAST ULTRASOUND ANALYSIS
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
import io
import base64

from PIL import Image
from huggingface_hub import hf_hub_download

from torchvision import transforms

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MammoSense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CONFIGURATION
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background:
        radial-gradient(
            circle at 10% 10%,
            rgba(220, 38, 127, 0.10),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 80%,
            rgba(59, 130, 246, 0.08),
            transparent 35%
        ),
        #f8fafc;
    }

    /* Header */

    .main-header {
        padding: 25px 0 10px 0;
    }

    .brand {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #111827;
        line-height: 1;
    }

    .brand span {
        color: #db2777;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-top: 8px;
    }

    /* Cards */

    .card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(226,232,240,0.9);
        border-radius: 20px;
        padding: 24px;
        box-shadow:
            0 10px 35px rgba(15,23,42,0.06);
        margin-bottom: 20px;
    }

    .result-card {
        background: white;
        border-radius: 22px;
        padding: 28px;
        border: 1px solid #e2e8f0;
        box-shadow:
            0 12px 35px rgba(15,23,42,0.07);
    }

    .section-title {
        font-size: 20px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 12px;
    }

    .small-text {
        color: #64748b;
        font-size: 13px;
    }

    /* Prediction */

    .prediction {
        font-size: 34px;
        font-weight: 800;
        margin: 4px 0;
    }

    .confidence {
        font-size: 18px;
        color: #64748b;
    }

    /* Pills */

    .pill {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        background: #fdf2f8;
        color: #be185d;
    }

    /* Disclaimer */

    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: 16px 18px;
        color: #9a3412;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 20px;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 30px 0;
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
    <div class="main-header">

        <div class="brand">
            Mammo<span>Sense</span>
        </div>

        <div class="subtitle">
            AI-assisted breast ultrasound analysis
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL ARCHITECTURE
# EXACTLY MATCHES TRAINING
# ============================================================

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
        ]

    def forward(
        self,
        x
    ):

        features = self.backbone(
            x
        )

        return self.head(
            features
        )


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

    if not isinstance(
        checkpoint,
        dict
    ):

        raise RuntimeError(
            "The downloaded file is not "
            "a valid MammoSense checkpoint."
        )

    if "model_state_dict" not in checkpoint:

        raise RuntimeError(
            "model_state_dict was not found "
            "inside the checkpoint."
        )

    num_classes = checkpoint.get(
        "num_classes",
        3
    )

    model = BUSIViT(
        num_classes=num_classes
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "model."
        ):

            key = key[
                len("model.") :
            ]

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    if missing:

        raise RuntimeError(
            "The checkpoint does not match "
            "the MammoSense architecture.\n\n"
            f"Missing keys: {missing[:10]}"
        )

    model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        checkpoint
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Initializing MammoSense AI..."
    ):

        model, checkpoint = load_model()

    model_status = True

except Exception as error:

    model_status = False

    st.error(
        "Unable to load the MammoSense model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose(

    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(

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

    ]

)


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model
    ):

        self.model = model

        self.activations = None
        self.gradients = None

        # ViT final block
        self.target_layer = (
            self.model
            .backbone
            .blocks[-1]
            .norm1
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
        image_tensor,
        target_class
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            image_tensor
        )

        score = logits[
            0,
            target_class
        ]

        score.backward()

        activations = self.activations

        gradients = self.gradients

        if activations is None or gradients is None:

            return None

        # ViT tensor:
        # [batch, tokens, channels]

        if activations.ndim != 3:

            return None

        activations = activations[
            0
        ]

        gradients = gradients[
            0
        ]

        # Remove CLS token

        if activations.shape[0] > 196:

            activations = activations[
                1:
            ]

            gradients = gradients[
                1:
            ]

        weights = gradients.mean(
            dim=0
        )

        cam = torch.sum(
            activations * weights,
            dim=1
        )

        cam = F.relu(
            cam
        )

        num_patches = cam.shape[0]

        grid_size = int(
            np.sqrt(
                num_patches
            )
        )

        if (
            grid_size
            * grid_size
            != num_patches
        ):

            return None

        cam = cam.reshape(
            grid_size,
            grid_size
        )

        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(
            cam,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam

    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam
):

    original = np.array(
        image.convert("RGB")
    )

    original = cv2.resize(
        original,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
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
        0.60,
        heatmap,
        0.40,
        0
    )

    return overlay


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    image,
    prediction,
    probabilities,
    confidence
):

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = styles[
        "Title"
    ]

    title_style.alignment = (
        TA_CENTER
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
            styles["Heading2"]
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
            "<b>AI Assessment</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(
            1,
            8
        )
    )

    data = [

        [
            "Category",
            "Result"
        ],

        [
            "Prediction",
            prediction
        ],

        [
            "Confidence",
            f"{confidence * 100:.2f}%"
        ],

        [
            "Model",
            "MammoSense ViT-Small"
        ],

        [
            "Architecture",
            "ViT-Small Patch16-224"
        ]

    ]

    table = Table(
        data,
        colWidths=[
            180,
            300
        ]
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#f1f5f9"
                    )
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#cbd5e1"
                    )
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
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
            ]
        )
    )

    story.append(
        table
    )

    story.append(
        Spacer(
            1,
            20
        )
    )

    story.append(
        Paragraph(
            "<b>Class Probabilities</b>",
            styles["Heading2"]
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

        probability_data.append(
            [
                name,
                f"{probabilities[i] * 100:.2f}%"
            ]
        )

    probability_table = Table(
        probability_data,
        colWidths=[
            180,
            300
        ]
    )

    probability_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#f1f5f9"
                    )
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#cbd5e1"
                    )
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        probability_table
    )

    story.append(
        Spacer(
            1,
            25
        )
    )

    story.append(
        Paragraph(
            "<b>Important Notice</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            "MammoSense is an AI research and "
            "decision-support prototype. The output "
            "is not a medical diagnosis and should "
            "not be used as a substitute for clinical "
            "assessment, radiological interpretation, "
            "biopsy or other appropriate medical "
            "evaluation.",
            styles["BodyText"]
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
            styles["BodyText"]
        )
    )

    document.build(
        story
    )

    buffer.seek(
        0
    )

    return buffer


# ============================================================
# MAIN UPLOAD CARD
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="section-title">
            Upload Ultrasound Image
        </div>

        <div class="small-text">
            Upload a breast ultrasound image
            in JPG, JPEG, PNG or TIFF format.
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

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [
            1,
            1
        ],
        gap="large"
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        st.markdown(
            """
            <div class="card">

                <div class="section-title">
                    Ultrasound Image
                </div>

            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing ultrasound..."
    ):

        tensor = transform(
            image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )

        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities_tensor = (
                torch.softmax(
                    logits,
                    dim=1
                )[0]
            )

        probabilities = (
            probabilities_tensor
            .detach()
            .cpu()
            .numpy()
        )

        prediction_index = int(
            np.argmax(
                probabilities
            )
        )

        prediction = CLASS_NAMES[
            prediction_index
        ]

        confidence = float(
            probabilities[
                prediction_index
            ]
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with right:

        st.markdown(
            """
            <div class="result-card">

                <div class="section-title">
                    AI Assessment
                </div>

            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="pill">
                AI CLASSIFICATION
            </div>

            <div class="prediction">
                {prediction}
            </div>

            <div class="confidence">
                Confidence: {confidence * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.markdown(
            "**Class probabilities**"
        )

        for i, name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                probabilities[i]
            )

            st.write(
                f"{name} — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

            <div class="section-title">
                Model Attention — Grad-CAM
            </div>

            <div class="small-text">
                The visualization highlights image regions
                that contributed to the model's prediction.
            </div>

        """,
        unsafe_allow_html=True
    )

    try:

        cam_model = GradCAM(
            model
        )

        cam_tensor = transform(
            image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )

        cam = cam_model.generate(
            cam_tensor,
            prediction_index
        )

        cam_model.close()

        if cam is not None:

            overlay = (
                create_gradcam_overlay(
                    image,
                    cam
                )
            )

            cam_col1, cam_col2 = st.columns(
                2
            )

            with cam_col1:

                st.image(
                    image,
                    caption="Original",
                    use_container_width=True
                )

            with cam_col2:

                st.image(
                    overlay,
                    caption="Grad-CAM",
                    use_container_width=True
                )

        else:

            st.info(
                "Grad-CAM could not be generated "
                "for this image."
            )

    except Exception as e:

        st.warning(
            "Grad-CAM could not be generated."
        )

        st.caption(
            str(e)
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # REPORT
    # ========================================================

    pdf = create_pdf_report(
        image=image,
        prediction=prediction,
        probabilities=probabilities,
        confidence=confidence
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    report_col1, report_col2, report_col3 = st.columns(
        [
            1,
            2,
            1
        ]
    )

    with report_col2:

        st.download_button(
            label="📄 Download Standard PDF Report",
            data=pdf,
            file_name="MammoSense_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <b>Medical disclaimer:</b>
    MammoSense is an AI research and decision-support
    prototype. Its predictions are not a medical diagnosis.
    Results should be interpreted by a qualified healthcare
    professional and should not replace clinical examination,
    radiological review, biopsy or other appropriate medical
    evaluation.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        MammoSense • AI-assisted breast ultrasound research

        <br><br>

        Powered by Vision Transformer technology

    </div>
    """,
    unsafe_allow_html=True
)
