# 🌿 Medicinal Plant Leaf Identifier

AI-powered plant leaf classification using a pre-trained **MobileNetV2**
deep learning model, built with **Streamlit**, **TensorFlow/Keras**,
**Pillow**, and **NumPy**.

Upload a photo of a leaf, and the app predicts which of 20 medicinal
plant species it most likely belongs to, along with a confidence score,
the top-5 most likely matches, and educational information about the
predicted plant.

---

## 📋 Table of Contents

1. [Project Description](#-project-description)
2. [Features](#-features)
3. [Model Information](#-model-information)
4. [Supported Plant Classes](#-supported-plant-classes-20)
5. [Project Structure](#-project-structure)
6. [Installation](#-installation)
7. [Running Locally](#-running-locally)
8. [Uploading to GitHub](#-uploading-to-github)
9. [Deploying on Streamlit Community Cloud](#-deploying-on-streamlit-community-cloud)
10. [Limitations](#-limitations)
11. [Disclaimer](#-disclaimer)

---

## 📖 Project Description

This project is a complete, deployment-ready web application for
identifying medicinal plant leaves from images. It uses an
**already-trained** MobileNetV2 convolutional neural network
(`best_mobilenetv2_correct_dataset.keras`) purely for **inference** —
the model is loaded as-is and is never retrained, fine-tuned, or
modified by this application.

---

## ✨ Features

- Upload a leaf image (JPG, JPEG, or PNG)
- Preview the uploaded image before prediction
- One-click **"🔍 Identify Plant"** prediction (no auto-predict spam)
- Displays the predicted plant name and confidence percentage
- Displays the **Top-5** most likely predictions with confidence bars
- Educational information for each of the 20 supported plant species
  (scientific name, description, common uses, traditional uses,
  identification characteristics)
- Clean, modern, plant-themed UI
- Robust error handling for missing model files, invalid images, and
  failed predictions
- Model is cached with `@st.cache_resource` so it loads only once per
  session
- No use of Colab/Kaggle/Drive paths — fully portable and
  deployment-ready

---

## 🧠 Model Information

| Property           | Value                                      |
|---------------------|---------------------------------------------|
| Architecture         | MobileNetV2                                |
| Input size           | 224 × 224 × 3                              |
| Number of classes    | 20                                          |
| Test accuracy        | 77.82%                                     |
| Model file           | `best_mobilenetv2_correct_dataset.keras`   |
| Framework            | TensorFlow / Keras                         |

The app applies MobileNetV2's standard `preprocess_input` function
(scales pixel values to the `[-1, 1]` range) to match the preprocessing
MobileNetV2 expects.

---

## 🌱 Supported Plant Classes (20)

The class index order below is fixed and **must not** be re-sorted
alphabetically, since it must match the order the model was trained on:

```
0  Aloevera
1  Amla
2  Amruta_Balli
3  Betel
4  Brahmi
5  Castor
6  Curry_Leaf
7  Doddapatre
8  Hibiscus
9  Insulin
10 Jasmine
11 Lemon
12 Mango
13 Mint
14 Neem
15 Pappaya
16 Pepper
17 Pomegranate
18 Rose
19 Tulasi
```

---

## 📁 Project Structure

```
plant-leaf-identifier/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md               # This file
├── .gitignore
│
├── model/
│   └── best_mobilenetv2_correct_dataset.keras   # <-- put your model file here
│
└── assets/
    └── (optional images/icons)
```

### 📌 Where to put your model file

Place your trained model file at:

```
plant-leaf-identifier/model/best_mobilenetv2_correct_dataset.keras
```

The app loads it using the **project-relative path**
`model/best_mobilenetv2_correct_dataset.keras`, so it will work
identically on your local machine and on Streamlit Community Cloud —
no Colab, Drive, or `/content` paths involved.

> ⚠️ **Note on file size:** Keras `.keras` model files (especially
> MobileNetV2-based ones) can be tens of megabytes. GitHub allows files
> up to 100 MB via normal `git push`. If your model file is larger than
> that, use [Git LFS](https://git-lfs.com/) to track it.

---

## 🔧 Installation

1. Clone or download this repository.
2. (Recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place your trained model file inside the `model/` folder as
   described above.

---

## ▶️ Running Locally

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will start a local development server and automatically open
the app in your browser (typically at `http://localhost:8501`).

---

## 🚀 Uploading to GitHub

1. Initialize a git repository (if you haven't already):

   ```bash
   git init
   ```

2. Add all project files:

   ```bash
   git add .
   ```

3. Commit the files:

   ```bash
   git commit -m "Initial commit: Medicinal Plant Leaf Identifier"
   ```

4. Create a new repository on GitHub (via the GitHub website), then
   link your local repo to it:

   ```bash
   git remote add origin https://github.com/<your-username>/plant-leaf-identifier.git
   ```

5. Push your code:

   ```bash
   git branch -M main
   git push -u origin main
   ```

> Make sure `best_mobilenetv2_correct_dataset.keras` is actually
> committed and pushed (check that it isn't accidentally excluded by
> `.gitignore` and that it's under GitHub's file size limits, or
> tracked via Git LFS if larger).

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push your project to a public (or Streamlit-accessible private)
   GitHub repository, as described above.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in
   with your GitHub account.
3. Click **"New app"**.
4. Select:
   - **Repository:** `<your-username>/plant-leaf-identifier`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"**.
6. Streamlit Cloud will automatically install the dependencies listed
   in `requirements.txt` and launch your app.
7. Once deployed, you'll get a public URL you can share
   (e.g., `https://<your-app-name>.streamlit.app`).

No additional configuration is required — the app uses only
project-relative paths and standard Python packages.

---

## ⚠️ Limitations

- The model has a measured **test accuracy of 77.82%** — predictions
  will sometimes be incorrect.
- Prediction quality depends heavily on image quality, lighting,
  background clutter, leaf orientation, and how closely the photo
  resembles the training data.
- The model only recognizes the 20 specific plant species listed
  above; leaves from other species will be forced into one of these
  20 categories (with the actual confidence typically low in such
  cases).
- This tool is not a substitute for expert botanical or medical
  identification.

---

## ⚠️ Disclaimer

Prediction is generated by an AI image classification model. Results
may vary depending on image quality, lighting, background, and leaf
appearance. This application is for **educational/informational
purposes only** and should **not** be considered medical advice. Plant
information — including traditional and medicinal uses — is provided
for educational purposes; always consult a qualified professional
before using any plant for medicinal purposes.
