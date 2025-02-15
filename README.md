# Pneumonia_Detection
Pneumonia Detection using Machine Learning Techniques and Explainable AI(XAI).
This project is a Pneumonia Detection System built using Deep Learning (DenseNet121) and Grad-CAM (Class Activation Map) for explainability.
It analyzes Chest X-ray images to detect Pneumonia, highlights affected lung regions, and generates a PDF report with all results.

**Key Features:**
✅ Pneumonia Detection with Confidence Percentage (Normal & Pneumonia).
✅ Severity Assessment: Normal, Mild, Severe.
✅ Treatment Suggestions and Home Remedies based on severity level.
✅ Grad-CAM Visualization to highlight pneumonia-affected lung regions.
✅ PDF Report Generation containing:

Uploaded Chest X-ray.
CAM Highlighted Image.
Confidence Levels, Threat Level, Treatment Advice, Home Remedies.

**🛠️ Technologies Used:**
Python
PyTorch (torch, torchvision)
OpenCV (cv2)
PIL (Pillow)
Streamlit (for User Interface)
FPDF (for PDF Report Generation)
