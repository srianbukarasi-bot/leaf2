"""
Medicinal Plant Leaf Identifier
--------------------------------
A Streamlit web application that uses a pre-trained EfficientNetV2B0
TensorFlow/Keras model to identify medicinal plant species from
leaf images.

IMPORTANT: This application performs INFERENCE ONLY using an
already-trained model. It does not train, fine-tune, or modify
the model in any way.
"""

import os
import io
import traceback
import gdown

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import tensorflow as tf

# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medicinal Plant Leaf Identifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# EFFICIENTNETV2B0 MODEL CONFIGURATION
# ============================================================
# The 91.63% test-accuracy model is hosted on Google Drive.
# Replace GOOGLE_DRIVE_FILE_ID with the ID from your Google Drive
# sharing link.

MODEL_PATH = os.path.join("model", "best_efficientnetv2b0.keras")
GOOGLE_DRIVE_FILE_ID = "1g_4icUWVwlNlBVBRUHJMHQwB8uQ3Vzqg"
IMG_SIZE = (224, 224)
TEST_ACCURACY = 91.63  # %

# ============================================================
# EXACT CLASS ORDER (DO NOT REORDER / DO NOT RESPELL)
# ============================================================

CLASS_NAMES = [
    "Aloevera",       # 0
    "Amla",           # 1
    "Amruta_Balli",   # 2
    "Betel",          # 3
    "Brahmi",         # 4
    "Castor",         # 5
    "Curry_Leaf",     # 6
    "Doddapatre",     # 7
    "Hibiscus",       # 8
    "Insulin",        # 9
    "Jasmine",        # 10
    "Lemon",          # 11
    "Mango",          # 12
    "Mint",           # 13
    "Neem",           # 14
    "Pappaya",        # 15
    "Pepper",         # 16
    "Pomegranate",    # 17
    "Rose",           # 18
    "Tulasi",         # 19
]

# ============================================================
# PLANT INFORMATION DATABASE
# ============================================================
# Educational information only. No medical claims are made.
# Wording deliberately uses "traditionally used for",
# "commonly used in", "may be used in traditional practices"
# to avoid implying proven medical efficacy.

PLANT_INFO = {
    "Aloevera": {
        "scientific_name": "Aloe vera",
        "description": (
            "A succulent plant with thick, fleshy, serrated green leaves "
            "that store a clear gel. Widely grown as a houseplant and in "
            "gardens across warm climates."
        ),
        "common_uses": [
            "Skincare products and cosmetics",
            "Soothing gels and moisturizers",
            "Ornamental potted plant",
        ],
        "traditional_uses": (
            "Traditionally used topically for minor skin soothing and in "
            "some traditional practices for digestive wellness. Not a "
            "substitute for medical treatment."
        ),
        "identification": (
            "Thick, pointed, fleshy leaves with serrated edges arranged in "
            "a rosette; leaves are grey-green and may have white flecks."
        ),
    },
    "Amla": {
        "scientific_name": "Phyllanthus emblica",
        "description": (
            "A deciduous tree known for its round, light-green, sour "
            "fruit (Indian gooseberry), with small feathery leaflets."
        ),
        "common_uses": [
            "Fruit consumed fresh, dried, or as pickle",
            "Hair oils and hair care products",
            "Ayurvedic formulations",
        ],
        "traditional_uses": (
            "Commonly used in traditional Ayurvedic practices as a source "
            "of Vitamin C and as a general wellness tonic."
        ),
        "identification": (
            "Small feathery, pinnate leaves resembling a fern frond; "
            "greenish-yellow round fruits with visible segment lines."
        ),
    },
    "Amruta_Balli": {
        "scientific_name": "Tinospora cordifolia",
        "description": (
            "A climbing shrub, also known as Guduchi, with heart-shaped "
            "leaves and a distinctive corky, warty stem."
        ),
        "common_uses": [
            "Herbal supplements and tonics",
            "Traditional Ayurvedic decoctions",
        ],
        "traditional_uses": (
            "May be used in traditional practices as a general immunity "
            "and wellness supplement."
        ),
        "identification": (
            "Heart-shaped leaves similar to betel leaves; thin climbing "
            "vine with a rough, warty bark."
        ),
    },
    "Betel": {
        "scientific_name": "Piper betle",
        "description": (
            "A vine with glossy, heart-shaped leaves belonging to the "
            "pepper family, commonly cultivated in South and Southeast Asia."
        ),
        "common_uses": [
            "Chewed with areca nut in some cultural traditions",
            "Used in religious and cultural ceremonies",
        ],
        "traditional_uses": (
            "Traditionally used for fresh breath and digestive comfort in "
            "some cultural practices."
        ),
        "identification": (
            "Glossy, heart-shaped leaves with a smooth, waxy texture and "
            "prominent veins radiating from the base."
        ),
    },
    "Brahmi": {
        "scientific_name": "Bacopa monnieri",
        "description": (
            "A small creeping herb with succulent, oblong leaves, often "
            "found growing in wet, marshy areas."
        ),
        "common_uses": [
            "Herbal supplements",
            "Traditional Ayurvedic preparations",
        ],
        "traditional_uses": (
            "Traditionally used in Ayurvedic practices in connection with "
            "memory and mental clarity support."
        ),
        "identification": (
            "Small, succulent, oval-shaped leaves arranged oppositely on "
            "creeping stems; tiny white or pale purple flowers."
        ),
    },
    "Castor": {
        "scientific_name": "Ricinus communis",
        "description": (
            "A fast-growing shrub with large, deeply lobed, star-shaped "
            "leaves, cultivated for its seed oil."
        ),
        "common_uses": [
            "Source of castor oil for industrial and cosmetic use",
            "Ornamental garden plant",
        ],
        "traditional_uses": (
            "Traditionally used topically in some practices for skin and "
            "hair care. Seeds are toxic if ingested and should never be "
            "consumed."
        ),
        "identification": (
            "Large, palmate leaves with 5–11 deep pointed lobes; reddish "
            "or green stems and spiky seed capsules."
        ),
    },
    "Curry_Leaf": {
        "scientific_name": "Murraya koenigii",
        "description": (
            "A small aromatic tree whose compound leaves are a staple "
            "flavoring herb in South Asian cooking."
        ),
        "common_uses": [
            "Culinary seasoning in curries and tempering",
            "Hair oil formulations",
        ],
        "traditional_uses": (
            "Commonly used in traditional practices as a digestive aid "
            "when included in cooking."
        ),
        "identification": (
            "Compound leaves with 11–21 small, glossy, aromatic leaflets "
            "arranged along a central stem."
        ),
    },
    "Doddapatre": {
        "scientific_name": "Plectranthus amboinicus",
        "description": (
            "A succulent, aromatic herb (also called Indian borage) with "
            "thick, fuzzy, fragrant leaves."
        ),
        "common_uses": [
            "Culinary herb in some regional cuisines",
            "Home remedies preparations",
        ],
        "traditional_uses": (
            "Traditionally used for soothing throat and respiratory "
            "comfort in some home remedy practices."
        ),
        "identification": (
            "Thick, fleshy, rounded leaves with a fuzzy texture and "
            "scalloped edges; strong pungent aroma when crushed."
        ),
    },
    "Hibiscus": {
        "scientific_name": "Hibiscus rosa-sinensis",
        "description": (
            "A flowering shrub known for its large, showy, trumpet-shaped "
            "flowers and glossy, serrated leaves."
        ),
        "common_uses": [
            "Ornamental garden and hedge plant",
            "Herbal teas made from petals",
            "Hair care preparations",
        ],
        "traditional_uses": (
            "Traditionally used in hair oils and herbal teas in various "
            "cultural practices."
        ),
        "identification": (
            "Glossy, dark green, serrated oval leaves; large colorful "
            "trumpet-shaped flowers with a prominent central stamen."
        ),
    },
    "Insulin": {
        "scientific_name": "Costus igneus",
        "description": (
            "Also called the 'insulin plant,' a small perennial herb with "
            "spirally arranged, glossy green leaves."
        ),
        "common_uses": [
            "Ornamental ground cover",
            "Traditional herbal preparations",
        ],
        "traditional_uses": (
            "Traditionally used in some folk practices related to blood "
            "sugar wellness; this is not a proven medical treatment and "
            "should never replace prescribed diabetes care."
        ),
        "identification": (
            "Broad, glossy, spirally arranged leaves along a fleshy stem; "
            "leaves have a slightly waxy surface."
        ),
    },
    "Jasmine": {
        "scientific_name": "Jasminum spp.",
        "description": (
            "A climbing or shrubby plant prized for its small, highly "
            "fragrant white flowers and dark green leaves."
        ),
        "common_uses": [
            "Perfumery and fragrance production",
            "Ornamental garden plant",
            "Flavoring for jasmine tea",
        ],
        "traditional_uses": (
            "Commonly used in traditional aromatherapy practices for "
            "relaxation."
        ),
        "identification": (
            "Small, opposite, dark green leaflets; star-shaped, highly "
            "fragrant white flowers, often blooming at night."
        ),
    },
    "Lemon": {
        "scientific_name": "Citrus limon",
        "description": (
            "A small evergreen tree bearing oval, bright yellow citrus "
            "fruit and glossy aromatic leaves."
        ),
        "common_uses": [
            "Culinary use of fruit and zest",
            "Essential oil extraction",
            "Cleaning and household products",
        ],
        "traditional_uses": (
            "Traditionally used for its Vitamin C content and as a "
            "flavoring agent in traditional beverages."
        ),
        "identification": (
            "Glossy, elliptical, aromatic leaves with finely toothed "
            "edges; thorny branches and fragrant white flowers."
        ),
    },
    "Mango": {
        "scientific_name": "Mangifera indica",
        "description": (
            "A large tropical tree bearing sweet, fleshy fruit and long, "
            "leathery, dark green leaves."
        ),
        "common_uses": [
            "Fruit consumption fresh or processed",
            "Wood used in furniture and religious rituals",
        ],
        "traditional_uses": (
            "Leaves are traditionally used in some cultural and religious "
            "practices, including decorative garlands."
        ),
        "identification": (
            "Long, leathery, lance-shaped leaves that are reddish when "
            "young and dark green when mature; prominent midrib."
        ),
    },
    "Mint": {
        "scientific_name": "Mentha spp.",
        "description": (
            "A fast-spreading aromatic herb with bright green, textured "
            "leaves and a cooling scent."
        ),
        "common_uses": [
            "Culinary herb and garnish",
            "Flavoring for beverages and confectionery",
            "Essential oil for aromatherapy",
        ],
        "traditional_uses": (
            "Traditionally used for digestive comfort and refreshing "
            "breath in various home remedy practices."
        ),
        "identification": (
            "Bright green, textured, toothed-edge leaves arranged "
            "oppositely on square stems; strong cooling aroma."
        ),
    },
    "Neem": {
        "scientific_name": "Azadirachta indica",
        "description": (
            "A fast-growing evergreen tree with pinnate leaves composed of "
            "serrated leaflets, well known in traditional practices."
        ),
        "common_uses": [
            "Natural pesticide formulations",
            "Skincare and haircare products",
            "Traditional oral care items (neem twigs)",
        ],
        "traditional_uses": (
            "Traditionally used in skincare and haircare practices for "
            "its cleansing properties."
        ),
        "identification": (
            "Pinnate compound leaves with 8–19 serrated, curved leaflets; "
            "small white flowers and olive-like fruit."
        ),
    },
    "Pappaya": {
        "scientific_name": "Carica papaya",
        "description": (
            "A tropical tree-like plant with a single stem and large, "
            "deeply lobed leaves, bearing oblong orange fruit."
        ),
        "common_uses": [
            "Fruit consumed fresh or in salads",
            "Papain enzyme extraction for food processing",
        ],
        "traditional_uses": (
            "Leaves are traditionally used in some folk practices, often "
            "prepared as extracts; always consult a professional before "
            "use."
        ),
        "identification": (
            "Very large, deeply lobed palmate leaves on long hollow "
            "petioles atop a tall, unbranched trunk."
        ),
    },
    "Pepper": {
        "scientific_name": "Piper nigrum",
        "description": (
            "A woody climbing vine cultivated for its berries, which are "
            "dried and used as black pepper spice."
        ),
        "common_uses": [
            "Culinary spice (black, white, and green pepper)",
            "Flavoring in traditional remedies",
        ],
        "traditional_uses": (
            "Commonly used in traditional practices as a digestive "
            "stimulant when included in food."
        ),
        "identification": (
            "Glossy, broad, heart-shaped to oval leaves with prominent "
            "veins; small clustered berries on hanging spikes."
        ),
    },
    "Pomegranate": {
        "scientific_name": "Punica granatum",
        "description": (
            "A deciduous shrub or small tree bearing round fruit filled "
            "with juicy, jewel-like seeds, and narrow glossy leaves."
        ),
        "common_uses": [
            "Fruit consumption and juice production",
            "Ornamental garden plant",
        ],
        "traditional_uses": (
            "Peel and leaves are traditionally used in some folk "
            "preparations; the fruit is widely valued for its nutritional "
            "content."
        ),
        "identification": (
            "Narrow, glossy, lance-shaped leaves; bright orange-red "
            "tubular flowers and round, leathery-skinned fruit."
        ),
    },
    "Rose": {
        "scientific_name": "Rosa spp.",
        "description": (
            "A flowering shrub known worldwide for its fragrant, layered "
            "petal blooms and thorny stems."
        ),
        "common_uses": [
            "Ornamental garden plant",
            "Perfumery and rose water production",
            "Culinary flavoring (rose syrup, petals)",
        ],
        "traditional_uses": (
            "Traditionally used in skincare preparations such as rose "
            "water for its soothing fragrance."
        ),
        "identification": (
            "Compound leaves with 3–7 serrated leaflets; thorny stems and "
            "layered, fragrant flowers in many colors."
        ),
    },
    "Tulasi": {
        "scientific_name": "Ocimum tenuiflorum (Holy Basil)",
        "description": (
            "A sacred aromatic shrub widely grown in Indian households, "
            "with small, fragrant, slightly serrated leaves."
        ),
        "common_uses": [
            "Religious and cultural rituals",
            "Herbal teas and infusions",
        ],
        "traditional_uses": (
            "Traditionally used in Ayurvedic practices for respiratory "
            "and general wellness support."
        ),
        "identification": (
            "Small, aromatic, ovate leaves with slightly serrated edges; "
            "purplish stems and spike-like flower clusters."
        ),
    },
}


# ============================================================
# MODEL LOADING (CACHED)
# ============================================================

@st.cache_resource(show_spinner="Loading EfficientNetV2B0 model...")
def load_model():
    """
    Download the 91.63% EfficientNetV2B0 model from Google Drive
    if it is not already present, then load it.

    The model is inference-only. It is not retrained or modified.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        if GOOGLE_DRIVE_FILE_ID == "PASTE_YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
            raise FileNotFoundError(
                "Google Drive File ID has not been configured. "
                "Set GOOGLE_DRIVE_FILE_ID in app.py."
            )

        try:
            gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=MODEL_PATH,
                quiet=False,
               
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not download the EfficientNetV2B0 model from Google Drive: {e}"
            )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model download failed. Expected model at '{MODEL_PATH}'."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into the format expected by the saved
    EfficientNetV2B0 model.

    IMPORTANT:
    EfficientNetV2B0 in this saved model contains its own internal
    Rescaling/Normalization layers. Therefore we DO NOT apply
    EfficientNetV2B0 preprocess_input here.
    """
    image = pil_image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ============================================================
# PREDICTION
# ============================================================

def predict(model, pil_image: Image.Image):
    """
    Run inference on the uploaded image and return:
      - top_1 (class_name, confidence)
      - top_5 list of (class_name, confidence)
    """
    processed = preprocess_image(pil_image)
    predictions = model.predict(processed, verbose=0)[0]  # shape: (20,)

    top5_idx = np.argsort(predictions)[::-1][:5]
    top5 = [(CLASS_NAMES[i], float(predictions[i]) * 100.0) for i in top5_idx]

    top1_name, top1_confidence = top5[0]
    return top1_name, top1_confidence, top5


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4e7a4e;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }
    .result-card {
        background-color: #f1f8f2;
        border: 1px solid #cfe8d3;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .pred-bar-label {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2e5233;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #5d4a00;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown('<p class="main-header">🌿 Medicinal Plant Leaf Identifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered plant leaf classification using EfficientNetV2B0</p>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ About this project", expanded=False):
    st.write(
        "This application uses a trained **EfficientNetV2B0** deep learning model "
        f"to identify **{len(CLASS_NAMES)} medicinal plant species** from a "
        "leaf photo. Upload a clear image of a single leaf, and the model "
        "will predict the most likely plant species along with its "
        "confidence score. The model was trained and evaluated to a test "
        f"accuracy of **{TEST_ACCURACY}%**."
    )

st.divider()

# ============================================================
# LOAD MODEL (WITH ERROR HANDLING)
# ============================================================

model = None
model_load_error = None

try:
    model = load_model()
except FileNotFoundError as e:
    model_load_error = str(e)
except Exception as e:
    model_load_error = (
        "An unexpected error occurred while loading the model:\n\n"
        f"{e}"
    )

if model_load_error:
    st.error(f"🚫 Model could not be loaded.\n\n{model_load_error}")
    st.info(
        "Set `GOOGLE_DRIVE_FILE_ID` in `app.py` to the ID from your Google Drive "
        "sharing link, then restart the app."
    )
    st.stop()

# ============================================================
# LAYOUT: UPLOAD + PREVIEW
# ============================================================

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📤 Upload a Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    pil_image = None
    image_error = None

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            pil_image = Image.open(io.BytesIO(file_bytes))
            pil_image.verify()  # sanity check the image is not corrupted
            # Re-open after verify() since verify() invalidates the file pointer
            pil_image = Image.open(io.BytesIO(file_bytes))
        except UnidentifiedImageError:
            image_error = "The uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file."
        except Exception as e:
            image_error = f"Could not read the uploaded image: {e}"

    if image_error:
        st.error(f"⚠️ {image_error}")

    if pil_image is not None and not image_error:
        st.image(pil_image, caption="Uploaded Leaf Image", use_container_width=True)

    predict_clicked = False
    if pil_image is not None and not image_error:
        predict_clicked = st.button("🔍 Identify Plant", type="primary", use_container_width=True)

with col_right:
    st.subheader("🌿 Prediction Result")

    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None

    if pil_image is not None and not image_error and predict_clicked:
        with st.spinner("Analyzing leaf image..."):
            try:
                top1_name, top1_conf, top5 = predict(model, pil_image)
                st.session_state.last_prediction = {
                    "top1_name": top1_name,
                    "top1_conf": top1_conf,
                    "top5": top5,
                }
            except Exception:
                st.session_state.last_prediction = None
                st.error(
                    "⚠️ Something went wrong while running the prediction. "
                    "Please try a different image."
                )
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

    result = st.session_state.last_prediction

    if result is None:
        st.info("Upload an image and click **Identify Plant** to see results here.")
    else:
        top1_name = result["top1_name"]
        top1_conf = result["top1_conf"]
        top5 = result["top5"]

        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:1.1rem; color:#555;">🌿 Predicted Plant:</div>
                <div style="font-size:1.8rem; font-weight:700; color:#1b5e20;">{top1_name.replace('_', ' ')}</div>
                <div style="font-size:1.0rem; color:#555; margin-top:0.6rem;">Confidence:</div>
                <div style="font-size:1.4rem; font-weight:600; color:#2e7d32;">{top1_conf:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Top 5 Predictions**")
        for rank, (name, conf) in enumerate(top5, start=1):
            display_name = name.replace("_", " ")
            st.markdown(f'<span class="pred-bar-label">{rank}. {display_name} — {conf:.2f}%</span>', unsafe_allow_html=True)
            st.progress(min(max(conf / 100.0, 0.0), 1.0))

st.divider()

# ============================================================
# PLANT INFORMATION SECTION
# ============================================================

if st.session_state.get("last_prediction"):
    predicted_class = st.session_state.last_prediction["top1_name"]
    info = PLANT_INFO.get(predicted_class)

    if info:
        st.subheader(f"📖 About {predicted_class.replace('_', ' ')}")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown(f"**Scientific name:** *{info['scientific_name']}*")
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Identification characteristics:** {info['identification']}")
        with c2:
            st.markdown("**Common uses:**")
            for use in info["common_uses"]:
                st.markdown(f"- {use}")
            st.markdown(f"**Traditional/medicinal uses:** {info['traditional_uses']}")

st.divider()

with st.expander("🌱 Browse all 20 supported plant species"):
    for cls in CLASS_NAMES:
        info = PLANT_INFO[cls]
        st.markdown(f"### {cls.replace('_', ' ')}")
        st.markdown(f"*{info['scientific_name']}*")
        st.markdown(info["description"])
        st.markdown(f"**Traditional/medicinal uses:** {info['traditional_uses']}")
        st.markdown("---")

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    f"""
    <div class="disclaimer-box">
    <strong>⚠️ Disclaimer:</strong> Prediction is generated by an AI image
    classification model with a measured test accuracy of {TEST_ACCURACY}%.
    Results may vary depending on image quality, lighting, background, and
    leaf appearance — no prediction is guaranteed to be correct. This
    application is for educational/informational purposes only. Plant
    information, including traditional and medicinal uses, is provided for
    educational purposes and should <strong>not</strong> be considered
    medical advice. Always consult a qualified professional (botanist,
    doctor, or herbalist) before using any plant for medicinal purposes.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Built with Streamlit, TensorFlow/Keras, and EfficientNetV2B0.")
