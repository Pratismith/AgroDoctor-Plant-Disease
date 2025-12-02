🌱 AgroDoctor: AI-Powered Plant Disease Detection System

📌 Project Overview

AgroDoctor is an end-to-end Computer Vision project designed to detect plant leaf diseases. It bridges the gap between traditional Image Processing (OpenCV) and modern Deep Learning (ResNet18) to provide accurate diagnoses accessible via a user-friendly Streamlit Web App.

This project demonstrates the full lifecycle of an AI solution: Data Engineering → Model Training (Cloud/GPU) → Model Deployment (Local Web App).

🏗️ System Architecture

The project consists of three core modules. You can explain these sections directly during your presentation.

1️⃣ Module 1: Preprocessing & Computer Vision (OpenCV)

Before the AI analyzes the image, we use classical image processing to visualize features and "clean" the data. This demonstrates an understanding of fundamental vision techniques.

Gaussian Blur: Removes high-frequency noise from the leaf image.

Contrast Enhancement (HE): Uses Histogram Equalization on the Y-channel to make disease spots stand out against the leaf surface.

Canny Edge Detection: Identifies the structural boundaries of the leaf and lesions.

HSV Masking: A color-based segmentation technique used to isolate the "diseased" regions (usually brown/yellow) from the healthy green parts.

2️⃣ Module 2: Deep Learning Brain (ResNet18)

The core classification engine is built using PyTorch.

Model: ResNet18 (Residual Neural Network).

Technique: Transfer Learning.

Why? Training from scratch requires millions of images. We used a pre-trained ResNet (trained on ImageNet) and "fine-tuned" it for plant diseases.

Mechanism: We froze the early layers (which detect lines/curves) and only retrained the final fully connected layers to recognize specific diseases (Blight, Rust, etc.).

Training Environment: Google Colab (T4 GPU).

Data was automatically split into 80% Training and 20% Validation.

Achieved convergence over 10 epochs.

3️⃣ Module 3: Web Deployment (Streamlit)

The interface allows non-technical users (farmers/researchers) to use the AI.

Framework: Streamlit (Python-based web app).

Features:

Real-time image upload.

Side-by-side visualization of Original vs. Processed images.

Confidence Score: Shows how certain the AI is about the diagnosis.

Explainable AI (Grad-CAM): A visualization heatmap that shows where the model is looking to make a decision (e.g., highlighting the infected spots).

🚀 How to Run the Project

Prerequisites

Ensure you have Python installed. Install the dependencies using:

pip install streamlit opencv-python-headless torch torchvision numpy pillow matplotlib


Steps to Launch

Place Files: Ensure app.py, plant_disease_model.pth, and class_names.json are in the same folder.

Run Command:

streamlit run app.py


Browser: The app will open automatically at http://localhost:8501.

🎤 Interview / Presentation Script

Use these bullet points when describing your work:

Q: Why did you combine OpenCV and Deep Learning?

"I used OpenCV for the visual analysis pipeline to show how image features like edges and contrast are extracted. However, for the actual classification, Deep Learning (ResNet) is far more robust to variations in lighting and angle than manual thresholding."

Q: Why ResNet18?

"It's a perfect balance between performance and speed. It solves the 'vanishing gradient' problem using skip connections, making it highly effective for feature extraction without being too heavy for a web deployment."

Q: How did you handle the data?

"I sourced the PlantVillage dataset. I used a custom script in Google Colab to automatically handle the directory structure, split the data into training and validation sets, and normalize the images to match ResNet's input requirements (Mean/Std)."

Q: What is the 'Heatmap' (Grad-CAM)?

"It represents Explainable AI (XAI). In a real-world scenario, we don't just want a prediction; we want to know why. The heatmap highlights the pixels that activated the neural network the most, confirming that the model is actually looking at the disease spots and not just the background."
