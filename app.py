import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from fpdf import FPDF

# Load the trained DenseNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load('pneumonia_model.pth', map_location=device))
model.to(device)
model.eval()

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to convert grayscale to RGB
def convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


# Grad-CAM function (for DenseNet121 specifically)
class CAMExtractor:
    def __init__(self, model):
        self.model = model
        self.gradient = None
        self.feature = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.feature = output

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0]

        self.model.features[-1].register_forward_hook(forward_hook)
        self.model.features[-1].register_backward_hook(backward_hook)

    def forward_pass(self, x):
        return self.model(x)

    def get_cam(self):
        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
        for i in range(self.feature.shape[1]):
            self.feature[:, i, :, :] *= pooled_gradients[i]
        cam = self.feature.mean(dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = np.uint8(255 * cam)
        return cam


# Visualize CAM on image with heatmap
def visualize_cam_on_image(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    return blended


# Generate PDF Report
def generate_pdf(image_path, cam_image_path, confidence_pneumonia, confidence_normal, threat_level, suggestion, home_remedies):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Pneumonia Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Threat Level: {threat_level}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence (Pneumonia): {confidence_pneumonia:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Confidence (Normal): {confidence_normal:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Suggestion: {suggestion}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Home Remedies: {home_remedies}")

    pdf.ln(10)
    pdf.cell(200, 10, txt="Uploaded X-ray/CT Scan Image:", ln=True)
    pdf.image(image_path, x=10, y=pdf.get_y(), w=90)
    pdf.ln(60)

    pdf.cell(200, 10, txt="CAM Visualization (Highlighted Regions):", ln=True)
    pdf.image(cam_image_path, x=10, y=pdf.get_y(), w=90)

    report_path = "Pneumonia_Report.pdf"
    pdf.output(report_path)
    return report_path


# Streamlit App UI
st.title("Pneumonia Detection Using DenseNet121 and CAM")
uploaded_file = st.file_uploader("Upload Chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = convert_to_rgb(image)  # Convert grayscale to RGB if needed
    st.image(image, caption='Uploaded Image', use_container_width=True)

    image_path = "uploaded_image.png"
    image.save(image_path)

    image_tensor = transform(image).unsqueeze(0).to(device)

    # CAM Extraction
    cam_extractor = CAMExtractor(model)
    output = cam_extractor.forward_pass(image_tensor)

    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    confidence_pneumonia = probabilities[1].item() * 100
    confidence_normal = probabilities[0].item() * 100

    class_idx = output.argmax().item()

    # Generate CAM visualization
    model.zero_grad()
    output[0, class_idx].backward()
    cam = cam_extractor.get_cam()
    cam_image = visualize_cam_on_image(image_path, cam)
    cam_image_path = "cam_visualization.png"
    cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    st.image(cam_image, caption='Pneumonia Region Highlighted (CAM)', use_container_width=True)

    # Threat level, suggestion, and remedies based on pneumonia confidence
    if confidence_pneumonia < 40:
        threat_level = "No Pneumonia (Normal)"
        suggestion = "No specific treatment needed. Maintain a healthy lifestyle."
        home_remedies = "Stay hydrated, practice good hygiene, and maintain a balanced diet."
    elif confidence_pneumonia < 70:
        threat_level = "Mild Pneumonia"
        suggestion = "Curable with medication. Consult a general physician."
        home_remedies = "Drink warm fluids, rest well, steam inhalation, avoid cold air."
    else:
        threat_level = "Severe Pneumonia"
        suggestion = "Immediate hospitalization is advised."
        home_remedies = "Seek emergency medical attention. Home care is not sufficient."

    st.markdown(
        f"""
        ### **Result:**
        - **Threat Level:** {threat_level}
        - **Confidence (Pneumonia):** {confidence_pneumonia:.2f}%
        - **Confidence (Normal):** {confidence_normal:.2f}%
        - **Suggestion:** {suggestion}
        - **Home Remedies:** {home_remedies}
        """
    )

    if st.button("Download Report as PDF"):
        report_path = generate_pdf(image_path, cam_image_path, confidence_pneumonia, confidence_normal, threat_level, suggestion, home_remedies)
        with open(report_path, "rb") as file:
            st.download_button(label="Download Report", data=file, file_name="Pneumonia_Report.pdf", mime="application/pdf")