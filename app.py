# app.py
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import types

# Model loading function with caching
@st.cache_resource
def load_model():
    # Initialize model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 256)
    model.out = torch.nn.Linear(256, 4)

    # Define custom forward
    def new_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.out(x)
        return x

    model.forward = types.MethodType(new_forward, model)
    
    # Load weights
    checkpoint = torch.load("CXR_model_f1.pth", map_location="cpu")
    model_state_dict = {k.replace("model.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
idx_to_label = {
    0: "Normal",
    1: "Viral Pneumonia",
    2: "Bacterial Pneumonia",
    3: "Covid"
}

# Streamlit UI
st.title("X-ray Image Classification")
st.write("Upload a chest X-ray image for classification")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess and predict
    with st.spinner('Analyzing image...'):
        img_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            model = load_model()
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        predicted_class = torch.argmax(probabilities).item()
        highest_prob = probabilities[predicted_class].item()

    # Display results
    st.success("Analysis Complete!")
    st.header(f"Prediction: {idx_to_label[predicted_class]}")
    st.subheader(f"Confidence: {highest_prob * 100:.2f}%")

    # Optional: Show all probabilities
    with st.expander("Show detailed probabilities"):
        for idx, prob in enumerate(probabilities):
            st.write(f"{idx_to_label[idx]}: {prob.item() * 100:.2f}%")