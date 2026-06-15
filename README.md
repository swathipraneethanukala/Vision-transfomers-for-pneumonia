# Chest X-Ray Pneumonia Detection (ViT)
This repository contains a full-stack web application that uses a fine-tuned **Vision Transformer (ViT)** model to detect pneumonia from chest X-ray images. 

The project uses a **Hugging Face ViT backend (`google/vit-base-patch16-224`)** wrapped in a Flask API to serve live predictions.Unlike traditional Convolutional Neural Networks (CNNs) that focus only on local pixel regions, the Vision Transformer uses a self-attention mechanism to analyze the entire X-ray image at once.This allows it to catch global pattern changes and long-range dependencies that might indicate infection across the lungs.

---

## 🚀 Key Features
***Global Context Attention:** Leverages a pre-trained ViT backbone fine-tuned on chest X-ray data, capturing complex patterns that span across the entire lung layout.
* **Real-Time Web UI:** Features a simple, clean drag-and-drop frontend (HTML/CSS/JS) that interacts asynchronously with the Python backend via API fetch requests.
* **Strict Weight Alignment:** The backend script loads the custom fine-tuned weights (`.pth` state dictionary) with validation mapping to ensure flawless layer alignment with the base Transformer model.
* **Standardized Preprocessing:** Preprocesses raw uploaded images into matching $224 \times 224$ tensors with specialized ImageNet normalization values to optimize classification reliability.

---

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch & Hugging Face Transformers (`ViTForImageClassification`)
* **Backend Core:** Flask, Flask-CORS (Cross-Origin Resource Sharing)
* **Frontend UI:** Vanilla HTML5, CSS3, JavaScript
* **Image Processing:** PIL (Pillow), Torchvision

---

## 📂 Repository Structure

```text
Vision_Transformers/
│
├── backend/
│   ├── app.py                      # Flask Server API & Hugging Face ViT Pipeline
│   └── best_vit_pneumonia_model.pth # Fine-tuned PyTorch weight file
│
└── frontend/
    ├── index.html                  # Main web dashboard interface
    ├── style.css                   # Responsive UI layouts and styling
    └── script.js                   # Handles image upload and async API calls
