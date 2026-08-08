import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download


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

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 1.5rem 2rem;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #172554 0%,
        #1e3a8a 55%,
        #2563eb 100%
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    opacity: 0.9;
}

.result-card {
    padding: 22px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 18px;
}

.disclaimer {
    padding: 18px;
    border-radius: 12px;
    background-color: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
}

.info-card {
    padding: 18px;
    border-radius: 14px;
    background-color: white;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

REPO_ID = "Makky07/MammoSense-breast-ultrasound"

# IMPORTANT:
# This is the exact filename currently stored on Hugging Face.
MODEL_FILENAME = "gaia_busi_vit_small (1).pt"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]

NUM_CLASSES = 3


# ============================================================
# EXACT MODEL ARCHITECTURE USED DURING TRAINING
# ============================================================

class BUSIViT(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        # EXACT BACKBONE USED DURING TRAINING
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )

        # EXACT CLASSIFICATION HEAD USED DURING TRAINING
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

    return model, checkpoint


# ============================================================
# IMAGE PREPROCESSING
# EXACTLY MATCHES eval_transform USED DURING TESTING
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

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return (
        predicted_class,
        probabilities.cpu().numpy()
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🩺 MammoSense</h1>

<p>
AI-assisted breast ultrasound image classification
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## MammoSense")

    st.markdown(
        "### AI Breast Ultrasound Analysis"
    )

    st.divider()

    st.markdown("### Model Information")

    st.write(
        "**Architecture:** ViT-Small Patch16-224"
    )

    st.write(
        "**Input size:** 224 × 224"
    )

    st.write(
        "**Classes:** 3"
    )

    st.write(
        "**Training dataset:** BUSI"
    )

    st.write(
        "**Framework:** PyTorch + timm"
    )

    st.divider()

    st.markdown("### Classes")

    st.write("🟢 Normal")
    st.write("🟡 Benign")
    st.write("🔴 Malignant")

    st.divider()

    st.caption(
        "MammoSense is an AI research and "
        "educational prototype."
    )


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    "## Breast Ultrasound Analysis"
)

st.write(
    "Upload a breast ultrasound image to obtain "
    "an AI-generated classification and probability "
    "distribution."
)


# ============================================================
# MODEL LOADING
# ============================================================

try:

    with st.spinner(
        "Loading MammoSense AI model..."
    ):

        model, checkpoint = load_model()

    st.success(
        "MammoSense model loaded successfully."
    )

except Exception as e:

    st.error(
        "Unable to load the MammoSense model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload breast ultrasound image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.divider()

    left, right = st.columns(
        [1, 1]
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### Uploaded Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### AI Classification"
        )

        with st.spinner(
            "Analyzing ultrasound image..."
        ):

            prediction, probabilities = predict(
                image,
                model
            )

        confidence = (
            float(probabilities.max())
            * 100
        )

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f"## {prediction}"
        )

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    st.divider()

    st.markdown(
        "## Prediction Probabilities"
    )

    cols = st.columns(3)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            float(probabilities[i])
            * 100
        )

        with cols[i]:

            st.markdown(
                f"### {class_name}"
            )

            st.progress(
                min(
                    probability / 100,
                    1.0
                )
            )

            st.write(
                f"**{probability:.2f}%**"
            )


    # ========================================================
    # MODEL INTERPRETATION
    # ========================================================

    st.divider()

    st.markdown(
        "## Analysis Summary"
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "**Predicted classification**"
        )

        st.markdown(
            f"### {prediction}"
        )

        st.write(
            f"The model assigned the highest "
            f"probability to **{prediction}**."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    with summary_col2:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "**Highest probability**"
        )

        st.markdown(
            f"### {confidence:.2f}%"
        )

        st.write(
            "This represents the model's "
            "predicted probability for its "
            "highest-scoring class."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.markdown("""
<div class="disclaimer">

### ⚠️ Important Medical Disclaimer

MammoSense is an experimental AI research prototype
and is **not a medical device or diagnostic system**.

Its predictions should not be used to diagnose,
exclude, or confirm breast cancer.

AI predictions may be incorrect, particularly for
images that differ from the training dataset.

Always seek evaluation by a qualified healthcare
professional and appropriate clinical imaging
assessment.

</div>
""", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MammoSense • AI-assisted breast ultrasound "
    "classification research prototype"
)

st.caption(
    "Model: ViT-Small Patch16-224 • "
    "Dataset: BUSI • 3-class classification"
)
