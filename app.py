
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MammoSense | Breast Ultrasound AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: white;
}

.mammo-header {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a5f
    );
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.mammo-title {
    color: white;
    font-size: 40px;
    font-weight: 800;
}

.mammo-subtitle {
    color: #cbd5e1;
    font-size: 17px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    text-align: center;
}

.result-title {
    color: #64748b;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    color: #0f172a;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.disclaimer {
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 18px;
    border-radius: 10px;
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HUGGING FACE MODEL
# ============================================================

REPO_ID = "Makky07/MammoSense-breast-ultrasound"
MODEL_FILENAME = "gaia_busi_vit_small.pt"


@st.cache_resource
def download_model():

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )

    return model_path


# ============================================================
# MODEL
# ============================================================

class MammoSenseViT(nn.Module):

    def __init__(
        self,
        architecture,
        num_classes
    ):

        super().__init__()

        self.backbone = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):

        return self.backbone(x)


@st.cache_resource
def load_model():

    model_path = download_model()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model = MammoSenseViT(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


model, checkpoint = load_model()


# ============================================================
# CHECKPOINT INFORMATION
# ============================================================

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]
architecture = checkpoint["architecture"]
num_classes = checkpoint["num_classes"]


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (image_size, image_size)
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
])


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="mammo-header">

<div class="mammo-title">
🩺 MammoSense
</div>

<div class="mammo-subtitle">
AI-assisted breast ultrasound image classification
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 MammoSense")

    st.caption(
        "Breast Ultrasound AI Research Platform"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🔬 Image Analysis",
            "🧠 Model Information",
            "ℹ️ About MammoSense"
        ]
    )

    st.divider()

    st.markdown("**Model Status**")

    st.success("Model loaded")

    st.caption(
        "ViT-Small • BUSI"
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if page == "🔬 Image Analysis":

    st.markdown(
        "## Breast Ultrasound Analysis"
    )

    st.write(
        "Upload a breast ultrasound image and "
        "MammoSense will classify it into one of "
        "three categories."
    )

    uploaded_file = st.file_uploader(
        "Upload ultrasound image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is None:

        st.info(
            "Upload an ultrasound image to begin."
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Architecture",
                "ViT-Small"
            )

        with c2:

            st.metric(
                "Classes",
                "3"
            )

        with c3:

            st.metric(
                "Input",
                f"{image_size} × {image_size}"
            )


    else:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        col1, col2 = st.columns(
            [1.1, 1]
        )


        with col1:

            st.markdown(
                "### Ultrasound Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        with col2:

            st.markdown(
                "### Analysis"
            )

            analyze = st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            )


            if analyze:

                with st.spinner(
                    "MammoSense is analyzing the image..."
                ):

                    x = transform(
                        image
                    ).unsqueeze(0)


                    with torch.no_grad():

                        output = model(x)

                        probabilities = F.softmax(
                            output,
                            dim=1
                        )[0]


                    prediction_index = torch.argmax(
                        probabilities
                    ).item()


                    prediction = class_names[
                        prediction_index
                    ]


                    confidence = probabilities[
                        prediction_index
                    ].item()


                st.session_state[
                    "prediction"
                ] = prediction

                st.session_state[
                    "confidence"
                ] = confidence

                st.session_state[
                    "probabilities"
                ] = probabilities.tolist()

                st.session_state[
                    "analyzed"
                ] = True


        if st.session_state.get(
            "analyzed",
            False
        ):

            prediction = st.session_state[
                "prediction"
            ]

            confidence = st.session_state[
                "confidence"
            ]

            probabilities = st.session_state[
                "probabilities"
            ]


            st.divider()

            st.markdown(
                "## MammoSense Result"
            )


            r1, r2, r3 = st.columns(3)


            with r1:

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="result-title">
                    Prediction
                    </div>

                    <div class="result-value">
                    {prediction}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r2:

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="result-title">
                    Confidence
                    </div>

                    <div class="result-value">
                    {confidence*100:.1f}%
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with r3:

                st.markdown(
                    f"""
                    <div class="result-card">

                    <div class="result-title">
                    Classes
                    </div>

                    <div class="result-value">
                    {num_classes}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # PROBABILITIES
            # =================================================

            st.markdown(
                "### Classification Probabilities"
            )


            for i, name in enumerate(
                class_names
            ):

                probability = probabilities[i]

                st.write(
                    f"**{name}** — "
                    f"{probability*100:.2f}%"
                )

                st.progress(
                    probability
                )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.markdown(
                """
                <div class="disclaimer">

                <strong>⚠️ Important Research Disclaimer</strong>

                <br><br>

                MammoSense is an artificial intelligence
                research prototype for breast ultrasound
                image classification.

                Its predictions must not be interpreted as
                a medical diagnosis or used as a substitute
                for assessment by a qualified healthcare
                professional.

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "🧠 Model Information":

    st.markdown(
        "## Model Information"
    )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Architecture",
            architecture
        )

    with b:

        st.metric(
            "Input Size",
            f"{image_size} × {image_size}"
        )

    with c:

        st.metric(
            "Classes",
            num_classes
        )


    st.markdown(
        "### Classification Categories"
    )

    for i, name in enumerate(
        class_names
    ):

        st.write(
            f"**{i} — {name}**"
        )


    st.markdown(
        "### Model Pipeline"
    )

    st.markdown(
        """
        **1. Image Upload**

        Breast ultrasound image is uploaded.

        **2. Preprocessing**

        Image is converted to RGB, resized to
        the model input size and normalized.

        **3. Vision Transformer**

        The image is processed by the trained
        ViT-Small model.

        **4. Classification**

        Probability scores are generated for:

        - Normal
        - Benign
        - Malignant

        **5. Prediction**

        The class with the highest probability
        becomes the model prediction.
        """
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About MammoSense":

    st.markdown(
        "## About MammoSense"
    )

    st.markdown(
        """
        ### MammoSense

        MammoSense is an artificial intelligence research
        platform focused on automated classification of
        breast ultrasound images.

        The current prototype uses a **Vision Transformer
        (ViT-Small)** model for three-class classification.

        ### Research Objectives

        - Breast ultrasound classification
        - Computer vision research
        - AI-assisted image analysis
        - Model interpretability
        - External dataset evaluation

        ### Current Classes

        **Normal • Benign • Malignant**

        ### Important

        MammoSense is currently a research prototype and
        has not been established as a clinical diagnostic
        tool.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    MammoSense • Breast Ultrasound AI Research Platform

    <br><br>

    Research prototype — not a medical diagnostic device.

    </div>
    """,
    unsafe_allow_html=True
)

