import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import traceback

# Safely import YOLO
try:
    from ultralytics import YOLO
except ImportError:
    st.info("The ultralytics package is not installed. Please install it with `pip install ultralytics`.")
    YOLO = None

# 1. YOLOv8 Segmentation and Cropping Functions
@st.cache_resource
def load_segmentation_model():
    try:
        if YOLO is None:
            return None
        if not os.path.exists('best.pt'):
            st.info("Model file 'best.pt' not found. Please make sure it's in the correct directory.")
            return None
        return YOLO('best.pt')
    except Exception as e:
        st.info("Segmentation model could not be loaded. Please check if the model file is available.")
        return None

def crop_segments(image_np, model, confidence_threshold=0.25):
    if model is None:
        return [], image_np
    
    try:
        results = model.predict(
            source=image_np,
            conf=confidence_threshold,
            imgsz=640
        )
        
        cropped_segments = []
        if results and results[0].masks is not None:
            masks = results[0].masks.data
            boxes = results[0].boxes.data
            masks_np = masks.cpu().numpy() if hasattr(masks, 'cpu') else masks
            
            for mask, box in zip(masks_np, boxes):
                try:
                    # Resize mask and create binary mask
                    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Apply mask and crop
                    x1, y1, x2, y2 = map(int, box[:4])
                    masked_segment = np.zeros_like(image_np)
                    for c in range(3):
                        masked_segment[:, :, c] = image_np[:, :, c] * binary_mask
                    cropped = masked_segment[y1:y2, x1:x2]
                    
                    if cropped.size > 0:
                        cropped_segments.append({
                            'image': Image.fromarray(cropped),
                            'class': results[0].names[int(box[5])] if int(box[5]) in results[0].names else 'Unknown',
                            'confidence': float(box[4])
                        })
                except Exception as e:
                    # Skip this segment if there's an error
                    pass
                    
        return cropped_segments, results[0].plot() if results else image_np
    except Exception as e:
        return [], image_np

# 2. MultiHeadViT Model Definition
class MultiHeadViT(nn.Module):
    def __init__(self, weights=ViT_B_16_Weights.IMAGENET1K_V1, dropout=0.2):
        super(MultiHeadViT, self).__init__()
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head1 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        self.head2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        out1 = self.head1(features)
        out2 = self.head2(features)
        return out1, out2

@st.cache_resource
def load_classification_model():
    try:
        model = MultiHeadViT(weights=None)
        model_path = "epoch_58.pth"
        
        if not os.path.exists(model_path):
            st.info(f"Classification model file '{model_path}' not found. Please make sure it's in the correct directory.")
            return None
            
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.info("Classification model could not be loaded. Please check if the model file is properly formatted.")
        return None

# 3. Preprocessing
vit_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Feature Extraction Functions
@st.cache_resource
def load_resnet_model():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.features = None
        
        # Register hook to capture features
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu()
    
    def extract_features(self, img_tensor):
        self.features = None  # Reset features
        with torch.no_grad():
            self.model(img_tensor)
        return self.features

def preprocess_image_for_resnet(image):
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def get_resnet50_layers():
    # Shortened list of key layers in ResNet-50 for better UI experience
    layers = [
        "conv1",
        "layer1.0.conv1",
        "layer1.2.conv3",
        "layer2.0.conv1",
        "layer2.3.conv3",
        "layer3.0.conv1",
        "layer3.5.conv3",
        "layer4.0.conv1",
        "layer4.2.conv3",
        "avgpool"
    ]
    
    # Group layers by category for better organization in the UI
    layer_groups = {
        "Initial Layers": ["conv1"],
        "Layer 1 (ResBlock 1)": ["layer1.0.conv1", "layer1.2.conv3"],
        "Layer 2 (ResBlock 2)": ["layer2.0.conv1", "layer2.3.conv3"],
        "Layer 3 (ResBlock 3)": ["layer3.0.conv1", "layer3.5.conv3"],
        "Layer 4 (ResBlock 4)": ["layer4.0.conv1", "layer4.2.conv3"],
        "Output Layers": ["avgpool"]
    }
    
    return layer_groups

def visualize_feature_maps(feature_maps, max_features=16):
    if feature_maps is None:
        return None
    
    # Handle different types of layers
    if len(feature_maps.shape) == 4:  # Conv layers: [batch, channels, height, width]
        feature_map = feature_maps[0]  # First batch
        num_channels = min(feature_map.shape[0], max_features)
        
        # Create figure to display feature maps in a grid
        fig, axes = plt.subplots(int(np.ceil(num_channels/4)), 4, figsize=(15, 3*int(np.ceil(num_channels/4))))
        
        if num_channels <= 4:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        for i in range(num_channels):
            feature = feature_map[i].numpy()
            # Normalize for better visualization
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
            axes[i].imshow(feature, cmap='viridis')
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        return fig
    
    elif len(feature_maps.shape) == 2:  # FC layers: [batch, features]
        feature_map = feature_maps[0]  # First batch
        num_features = min(feature_map.shape[0], max_features)
        
        # Create a bar plot for FC layer features
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(num_features), feature_map[:num_features].numpy())
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Activation Value')
        ax.set_title('FC Layer Activations')
        
        return fig
    
    else:  # For other layer types
        return None

# 5. Streamlit App
def page():
    st.title("Segmentation + MultiHeadViT Pipeline")
    st.write("1. YOLOv8 segments objects")
    st.write("2. MultiHeadViT classifies each segment")
    st.write("3. Feature analysis available")

    # Create a session state to track which segment is selected for feature analysis
    if 'show_feature_analysis' not in st.session_state:
        st.session_state.show_feature_analysis = False
        st.session_state.selected_segment_index = None

    if 'previous_file' not in st.session_state:
        st.session_state.previous_file = None
        
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)

    # Reset feature analysis when a new file is uploaded
    if uploaded_file is not None and st.session_state.previous_file != uploaded_file:
        st.session_state.show_feature_analysis = False
        st.session_state.selected_segment_index = None
        st.session_state.previous_file = uploaded_file

    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            # Load models
            seg_model = load_segmentation_model()
            if seg_model is None:
                st.info("Using direct classification without segmentation due to missing segmentation model.")
                cropped_segments = [{
                    'image': image,
                    'class': 'Full Image',
                    'confidence': 1.0
                }]
                seg_result = image_np
            else:
                # Segment and crop
                with st.spinner("Segmenting image..."):
                    cropped_segments, seg_result = crop_segments(image_np, seg_model, confidence_threshold)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(seg_result, caption="Segmentation Result", use_container_width=True)
            
            if cropped_segments:
                st.subheader("Cropped Segments Classification")
                
                # Load classification model
                vit_model = load_classification_model()
                
                if vit_model is None:
                    st.info("Classification model is not available. Showing segments without classification.")
                    for i, segment in enumerate(cropped_segments):
                        with st.expander(f"Segment {i+1}: {segment['class']} (Confidence: {segment['confidence']:.2f})", expanded=True):
                            st.image(segment['image'], caption="Cropped Segment", use_container_width=False, width=500)
                            if st.button(f"Feature Analysis", key=f"feature_btn_{i}"):
                                st.session_state.show_feature_analysis = True
                                st.session_state.selected_segment_index = i
                                st.rerun()
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    vit_model.to(device)
                    
                    for i, segment in enumerate(cropped_segments):
                        with st.expander(f"Segment {i+1}: {segment['class']} (Confidence: {segment['confidence']:.2f})", expanded=True):
                            col1, col2 = st.columns([3, 4])
                            
                            with col1:
                                st.image(segment['image'], caption="Cropped Segment", use_container_width=False, width=500)
                                # Feature Analysis Button
                                if st.button(f"Feature Analysis", key=f"feature_btn_{i}"):
                                    st.session_state.show_feature_analysis = True
                                    st.session_state.selected_segment_index = i
                                    st.rerun()
                            
                            with col2:
                                # Preprocess and predict
                                try:
                                    with st.spinner("Classifying..."):
                                        input_tensor = vit_preprocess(segment['image'])
                                        input_batch = input_tensor.unsqueeze(0).to(device)
                                        
                                        with torch.no_grad():
                                            out1, out2 = vit_model(input_batch)
                                        
                                        prob1 = torch.nn.functional.softmax(out1, dim=1)
                                        prob2 = torch.nn.functional.softmax(out2, dim=1)
                                        
                                        # Get class confidence percentages
                                        head1_astrocytes_conf = float(prob1[0][0]) * 100
                                        head1_cancerous_conf = float(prob1[0][1]) * 100
                                        head2_noncancerous_conf = float(prob2[0][0]) * 100
                                        head2_cancerous_conf = float(prob2[0][1]) * 100
                                        
                                        # Determine classifications
                                        head1_class = "Astrocytes" if prob1[0][0] > prob1[0][1] else "Cancerous"
                                        head2_class = "Non-Cancerous" if prob2[0][0] > prob2[0][1] else "Cancerous"
                                        
                                        # Display results in a structured format
                                        st.markdown("### Classification Results")
                                        
                                        # Object detection confidence
                                        st.markdown(f"**Detection Confidence:** {segment['confidence'] * 100:.2f}%")
                                        
                                        # Create metrics for classifications
                                        metrics_cols = st.columns(2)
                                        with metrics_cols[0]:
                                            st.metric(label="Head 1 Classification", value=head1_class)
                                        with metrics_cols[1]:
                                            st.metric(label="Head 2 Classification", value=head2_class)
                                        
                                        # Create a detailed table of confidences
                                        st.markdown("#### Confidence Scores")
                                        data = {
                                            "Head": ["Head 1", "Head 1", "Head 2", "Head 2"],
                                            "Class": ["Astrocytes", "Cancerous", "Non-Cancerous", "Cancerous"],
                                            "Confidence": [
                                                f"{head1_astrocytes_conf:.2f}%",
                                                f"{head1_cancerous_conf:.2f}%",
                                                f"{head2_noncancerous_conf:.2f}%",
                                                f"{head2_cancerous_conf:.2f}%"
                                            ]
                                        }
                                        
                                        # Progress bars for visual comparison
                                        st.markdown("##### Head 1")
                                        st.progress(float(head1_astrocytes_conf/100))
                                        st.caption(f"Astrocytes: {head1_astrocytes_conf:.2f}%")
                                        st.progress(float(head1_cancerous_conf/100))
                                        st.caption(f"Cancerous: {head1_cancerous_conf:.2f}%")
                                        
                                        st.markdown("##### Head 2")
                                        st.progress(float(head2_noncancerous_conf/100))
                                        st.caption(f"Non-Cancerous: {head2_noncancerous_conf:.2f}%")
                                        st.progress(float(head2_cancerous_conf/100))
                                        st.caption(f"Cancerous: {head2_cancerous_conf:.2f}%")
                                        
                                except Exception as e:
                                    st.info("Classification couldn't be performed on this segment.")
                                    st.error(f"Error: {str(e)}")
                
                # If feature analysis is triggered, show the feature analysis section
                if st.session_state.show_feature_analysis and st.session_state.selected_segment_index is not None:
                    st.markdown("---")
                    st.header("Feature Analysis")
                    
                    try:
                        # Validate the selected segment index
                        if st.session_state.selected_segment_index >= len(cropped_segments):
                            st.error("Invalid segment selection. Please try again.")
                            st.session_state.show_feature_analysis = False
                            st.session_state.selected_segment_index = None
                            st.rerun()
                        
                        selected_segment = cropped_segments[st.session_state.selected_segment_index]
                        
                        # Display the selected segment
                        st.subheader(f"Analyzing Segment {st.session_state.selected_segment_index + 1}")
                        st.image(selected_segment['image'], caption="Selected Segment", width=300)
                        
                        # Load ResNet model for feature extraction
                        with st.spinner("Loading ResNet model..."):
                            resnet_model = load_resnet_model()
                        
                        # Layer selection
                        layer_groups = get_resnet50_layers()
                        selected_group = st.selectbox("Layer Group", list(layer_groups.keys()))
                        selected_layer = st.selectbox("Layer", layer_groups[selected_group])
                        
                        # Number of feature maps to display
                        max_features = st.slider("Max Feature Maps to Display", 4, 64, 16)
                        
                        # Prepare image for model
                        img_tensor = preprocess_image_for_resnet(selected_segment['image'])
                        
                        # Extract features with better error handling
                        extractor = FeatureExtractor(resnet_model, selected_layer)
                        with st.spinner("Extracting features..."):
                            features = extractor.extract_features(img_tensor)
                        
                        # Visualize features
                        st.subheader(f"Feature Maps for layer: {selected_layer}")
                        
                        if features is not None:
                            # Display feature information
                            st.write(f"Feature shape: {features.shape}")
                            
                            # Visualize feature maps
                            with st.spinner("Generating visualization..."):
                                fig = visualize_feature_maps(features, max_features)
                            if fig is not None:
                                st.pyplot(fig)
                            else:
                                st.info("Could not generate visualization for this layer type.")
                            
                            # Display summary information
                            st.subheader("Layer Information")
                            st.write(f"Layer shape: {features.shape}")
                            
                            if len(features.shape) == 4:  # Conv layers
                                st.write(f"Number of channels: {features.shape[1]}")
                                st.write(f"Feature map size: {features.shape[2]}×{features.shape[3]}")
                            elif len(features.shape) == 2:  # FC layers
                                st.write(f"Number of features: {features.shape[1]}")
                        else:
                            st.info("Could not extract features from the selected layer.")
                    except Exception as e:
                        st.error(f"Error during feature analysis: {str(e)}")
                        st.code(traceback.format_exc())
                    
                    # Button to close feature analysis view
                    if st.button("Close Feature Analysis"):
                        st.session_state.show_feature_analysis = False
                        st.session_state.selected_segment_index = None
                        st.rerun()
            else:
                st.info("No segments were detected in the image. Try adjusting the confidence threshold.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.code(traceback.format_exc())
            st.info("There was an issue processing the image. Please try with a different image.")
    else:
        st.info("Please upload an image to see predictions.")

if __name__ == "__main__":
    page()