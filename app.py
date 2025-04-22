import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

# Set page config
st.set_page_config(
    page_title="YOLOv8 Instance Segmentation",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("YOLOv8 Instance Segmentation App")
st.write("Upload an image or video to perform instance segmentation using YOLOv8s model")

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.01)
model_type = st.sidebar.radio("Select Task", ["Image", "Video"])

# Load model (cache to avoid reloading)
@st.cache_resource
def load_model():
    try:
        model = YOLO('model.pt')  # Make sure model.pt is in your directory
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

model = load_model()

# Function to process image
def process_image(image, conf=confidence_threshold, iou=iou_threshold):
    if model is None:
        st.error("Model not loaded")
        return image
    
    # Run inference
    results = model.predict(image, conf=conf, iou=iou)
    
    # Plot results
    res_plotted = results[0].plot()
    return res_plotted

# Function to process video
def process_video(video_path, conf=confidence_threshold, iou=iou_threshold):
    if model is None:
        st.error("Model not loaded")
        return None
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temp file for output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output.name
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = model.predict(frame, conf=conf, iou=iou)
        res_plotted = results[0].plot()
        out.write(res_plotted)
        
        processed_frames += 1
        progress = int((processed_frames / total_frames) * 100)
        progress_bar.progress(min(progress, 100))
    
    cap.release()
    out.release()
    progress_bar.empty()
    
    return temp_output_path

# Main content
if model_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                # Convert PIL image to numpy array
                image_np = np.array(image)
                # Process image
                result_image = process_image(image_np)
                st.image(result_image, caption="Segmented Image", use_column_width=True)

elif model_type == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        
        if st.button("Process Video"):
            with st.spinner("Processing video..."):
                output_path = process_video(tfile.name)
                
                if output_path:
                    st.success("Video processing complete!")
                    st.video(output_path)
                    
                    # Clean up
                    os.unlink(tfile.name)
                    os.unlink(output_path)

# Footer
st.markdown("---")
st.markdown("YOLOv8 Instance Segmentation App | Made with Streamlit")