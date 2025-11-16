import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import random
import os
import io
from ultralytics import YOLO

# --- App Configuration ---
# Set the layout to be wide
st.set_page_config(layout="wide", page_title="Brain Tumor Detector")

# --- Mock Model Function ---
# This function simulates running your YOLOv11n.pt model.
# Replace this with your actual model loading and prediction code.
def mock_yolo_detection(pil_image, confidence_threshold):
    """
    MOCK FUNCTION: Simulates YOLO detection.
    
    Takes: A PIL Image
    Returns: A list of detection dictionaries, e.g.:
             [{"box": [x, y, w, h], "confidence": 0.95, "class": "tumor"}]
    """
    
    # Simulate that not all slices have a tumor
    if random.random() < 0.3: # 30% chance of no detection
        return []
        
    detections = []
    # Simulate 1 or 2 detections on a slice
    for _ in range(random.randint(1, 2)):
        # Generate a random box
        w, h = pil_image.size
        box_w = random.randint(int(w*0.1), int(w*0.3))
        box_h = random.randint(int(h*0.1), int(h*0.3))
        box_x = random.randint(int(w*0.2), int(w*0.7 - box_w))
        box_y = random.randint(int(h*0.2), int(h*0.7 - box_h))
        
        confidence = random.uniform(confidence_threshold, 0.99)
        
        if confidence >= confidence_threshold:
            detections.append({
                "box": [box_x, box_y, box_w, box_h],
                "confidence": confidence,
                "class": random.choice(["Glioma", "Meningioma", "Pituitary"])
            })
            
    return detections

# --- Real Model Function ---
def run_real_yolo_detection(pil_image, confidence_threshold):
    @st.cache_resource
    def load_model():
        return YOLO('best.pt')
    model = load_model()
       
    # 2. Run prediction
    results = model(pil_image, conf=confidence_threshold)
       
    # 3. Process results and return them in the same format as the mock function.
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            detections.append({
                "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
               "confidence": float(box.conf[0]),
               # --- MODIFICATION ---
               # Make sure to get the class name from your model
               "class": model.names[int(box.cls[0])]
                # ---
            })
    return detections

# --- Helper Function ---
def draw_boxes_on_image(pil_image, detections):
    """Draws bounding boxes on a PIL Image."""
    
    # Create a new image to draw on
    image_with_boxes = pil_image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    total_area_pixels = 0
    
    for det in detections:
        x, y, w, h = det['box']
        # Draw the rectangle
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        
        text = f"{det['class']}: {det['confidence']:.2f}"
        draw.text((x, y - 10), text, fill="red")
        
        # Calculate area
        total_area_pixels += (w * h)
        
    return image_with_boxes, total_area_pixels


# --- MLOps Feedback Function ---
def save_for_retraining(image_bytes, filename, notes):
    # Define the "staging" directory for new data
    STAGING_PATH = "new_data_for_training"
    
    if not os.path.exists(STAGING_PATH):
        os.makedirs(STAGING_PATH)
        
    save_filename = f"{os.path.splitext(filename)[0]}_review.png"
    notes_filename = f"{os.path.splitext(filename)[0]}_review.txt"
    
    try:
        # Save the image
        with open(os.path.join(STAGING_PATH, save_filename), "wb") as f:
            f.write(image_bytes)
            
        # Save the doctor's notes
        with open(os.path.join(STAGING_PATH, notes_filename), "w") as f:
            f.write(notes)
            
        return True
    except Exception as e:
        st.error(f"Error saving file for review: {e}")
        return False
        

# ==============================================================================
# --- STREAMLIT UI ---
# ==============================================================================

# --- Header ---
st.title("🧠 Brain Tumor AI Assistant")
st.markdown("Upload a series of 2D MRI slices to perform detection and quantitative assessment.")

# --- Sidebar (Controls) ---
with st.sidebar:
    st.header("Scan Parameters")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload MRI Scan Slices (PNG or JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    st.markdown("---")
    
    # Quantitative Assessment Inputs
    st.subheader("Quantitative Assessment")
    pixel_spacing = st.number_input("Pixel Spacing (mm/pixel)", min_value=0.1, value=0.5, step=0.01, help="The real-world size of a single pixel. Found in scan metadata.")
    
    st.markdown("---")
    
    # Model Parameters
    st.subheader("Model Settings")
    confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)


# --- Main Area (Results) ---
main_area, = st.tabs(["Scan Analysis"])

with main_area:
    if run_button and uploaded_files:
        st.header("Scan Analysis Results")
        max_pixel_area_found = 0
        max_area_slice_name = "N/A"
        
        with st.spinner("Running AI analysis on all slices..."):
            
            # Create columns for a cleaner layout
            col1, col2 = st.columns([1, 1])
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Load image
                image_bytes = uploaded_file.getvalue()
                pil_image = Image.open(io.BytesIO(image_bytes))

                # --- Run Detection ---
                detections = run_real_yolo_detection(pil_image, confidence_threshold)
                
                # Draw boxes on the image
                image_with_boxes, slice_area_pixels = draw_boxes_on_image(pil_image, detections)
                
                # --- MODIFICATION ---
                # Check if this slice has the new largest tumor area
                if slice_area_pixels > max_pixel_area_found:
                    max_pixel_area_found = slice_area_pixels
                    max_area_slice_name = uploaded_file.name
                # ---
                
                # --- Display Results ---
                # Alternate between columns
                target_col = col1 if i % 2 == 0 else col2 
                
                with target_col:
                    st.subheader(f"Slice: {uploaded_file.name}")
                    st.image(image_with_boxes, use_container_width=True)
                    
                    if detections:
                        # Report the types of tumors found on this slice
                        detected_classes = ", ".join(list(set([d['class'] for d in detections])))
                        st.caption(f"Detected: {detected_classes} | Total Area: {slice_area_pixels} px²")
                    else:
                        st.caption("No tumor detected on this slice.")
                        
                    # --- MLOps Feedback Loop ---
                    with st.expander("Review this detection (MLOps Feedback)"):
                        st.write("If this detection is incorrect, please flag it for review.")
                        
                        # Create a unique key for each widget
                        key_notes = f"notes_{uploaded_file.name}"
                        key_submit = f"submit_{uploaded_file.name}"
                        
                        notes = st.text_area("Doctor's Notes (e.g., 'Missed tumor' or 'False positive')", key=key_notes)
                        
                        if st.button("Submit for Review", key=key_submit):
                            if save_for_retraining(image_bytes, uploaded_file.name, notes):
                                st.success(f"Slice {uploaded_file.name} submitted for review. Thank you!")
                            else:
                                st.error("Could not save file for review.")
                    
                    st.markdown("---")

        # --- Final Report ---
        st.header("Quantitative Assessment Report")
        
        # 1. Convert max pixel area to real area (mm²)
        max_area_mm2 = (pixel_spacing ** 2) * max_pixel_area_found
        
        st.metric(
            label="**Estimated Maximum Tumor Area**",
            value=f"{max_area_mm2:.2f} mm²"
        )
        
        st.info(f"The largest tumor cross-section was found on slice **{max_area_slice_name}**. "
               f"Calculation based on a pixel spacing of {pixel_spacing}mm/px.")

    elif run_button:
        st.error("Please upload MRI slices in the sidebar to begin.")
    else:
        st.info("Upload your MRI scan slices and set the parameters in the sidebar, then click 'Run Analysis'.")