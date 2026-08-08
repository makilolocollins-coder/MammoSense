import streamlit as st
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files


# ============================================================
# MAMMOSENSE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MammoSense",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

REPO_ID = "Makky07/MammoSense-breast-ultrasound"

IMAGE_SIZE = 224
NUM_CLASSES = 3

CLASS_NAMES = [
    "Normal",
    "Benign",
    "Malignant"
]


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
    padding: 1.8rem 2rem;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #172554 0%,
        #1e3a8a 55%,
        #2563eb 100%
    );
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    font-size: 44px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.92;
}

.result-card {
    padding: 24px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 18px;
}

.info-card {
    padding: 20px;
    border-radius: 15px;
    background: white;
    border: 1px solid #e5e7eb;
}

.disclaimer {
    padding: 20px;
    border-radius: 14px;
    background-color: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# EXACT BUSIVIT MODEL USED DURING TRAINING
# ============================================================

class BUSIViT(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        # EXACT BACKBONE FROM TRAINING
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0
        )

        # EXACT CLASSIFICATION HEAD FROM TRAINING
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
# FIND AND DOWNLOAD MODEL FROM HUGGING FACE
# ============================================================

@st.cache_resource
def download_model():

    # Get the files currently inside the repository
    files = list_repo_files(
        repo_id=REPO_ID,
        repo_type="model"
    )

    # Find all .pt files
    model_files = [
        file
        for file in files
        if file.lower().endswith(".pt")
    ]

    if not model_files:

        raise FileNotFoundError(
            "No .pt model file was found in the "
            "MammoSense Hugging Face repository."
        )

    # Use the first .pt model found
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

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    # Create EXACT architecture used during training
    model = BUSIViT(
        num_classes=checkpoint["num_classes"]
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    # Evaluation mode
    model.eval()

    return model, checkpoint, model_filename


# ============================================================
# PREPROCESSING
# EXACTLY MATCHES TRAINING eval_transform
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

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        logits = model(
            image_tensor
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
        "### Breast Ultrasound AI"
    )

    st.divider()

    st.markdown(
        "### Model Information"
    )

    st.write(
        "**Architecture:** ViT-Small Patch16-224"
    )

    st.write(
        "**Input:** 224 × 224 pixels"
    )

    st.write(
        "**Classes:** 3"
    )

    st.write(
        "**Dataset:** BUSI"
    )

    st.write(
        "**Framework:** PyTorch + timm"
    )

    st.divider()

    st.markdown("### Classification")

    st.write("🟢 Normal")
    st.write("🟡 Benign")
    st.write("🔴 Malignant")

    st.divider()

    st.caption(
        "MammoSense is an experimental "
        "AI research prototype."
    )


# ============================================================
# MAIN INTRODUCTION
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
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Connecting to MammoSense AI model..."
    ):

        model, checkpoint, model_filename = load_model()

    st.success(
        f"MammoSense model loaded successfully: "
        f"{model_filename}"
    )

except Exception as e:

    st.error(
        "Unable to load the MammoSense model."
    )

    st.code(
        str(e),
        language="text"
    )

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

    image_column, result_column = st.columns(
        [1, 1]
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with image_column:

        st.markdown(
            "### Uploaded Ultrasound"
        )

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with result_column:

        st.markdown(
            "### AI Classification"
        )

        with st.spinner(
            "Analyzing image..."
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
            "Highest model probability",
            f"{confidence:.2f}%"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.divider()

    st.markdown(
        "## Prediction Probabilities"
    )

    probability_columns = st.columns(3)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            float(probabilities[i])
            * 100
        )

        with probability_columns[i]:

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
    # SUMMARY
    # ========================================================

    st.divider()

    st.markdown(
        "## Analysis Summary"
    )

    col1, col2 = st.columns(2)

    with col1:

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
            f"The model assigned its highest "
            f"probability to **{prediction}**."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div class="info-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "**Model probability**"
        )

        st.markdown(
            f"### {confidence:.2f}%"
        )

        st.write(
            "This is the model's probability "
            "for its highest-scoring class."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Model information"
):

    st.write(
        "**Model:** MammoSense BUSI ViT"
    )

    st.write(
        "**Architecture:** "
        "vit_small_patch16_224"
    )

    st.write(
        "**Input size:** 224 × 224"
    )

    st.write(
        "**Classes:** Normal, Benign, Malignant"
    )

    st.write(
        "**Training dataset:** BUSI "
        "(Breast Ultrasound Images)"
    )

    st.write(
        "**Image normalization:** "
        "ImageNet mean/std"
    )

    st.write(
        f"**Hugging Face repository:** "
        f"{REPO_ID}"
    )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.divider()

st.markdown("""
<div class="disclaimer">

<h3>⚠️ Important Medical Disclaimer</h3>

MammoSense is an experimental AI research prototype
and is <b>not a medical device or diagnostic system</b>.

Its predictions should not be used to diagnose,
exclude, or confirm breast cancer.

AI predictions can be incorrect, particularly when
images differ from the training data.

Clinical assessment by a qualified healthcare
professional is required for diagnosis.

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
    "ViT-Small Patch16-224 • BUSI • "
    "Normal / Benign / Malignant"
)
