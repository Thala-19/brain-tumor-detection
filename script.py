import streamlit as st
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageDraw
import random
import os
import io

# --- App Configuration ---
# Set the layout to be wide
st.set_page_config(layout="wide", page_title="Brain Tumor Detector")

# --- Real Model Function ---
def run_real_yolo_detection(pil_image, confidence_threshold):
    import numpy as np
    from math import ceil

    @st.cache_resource
    def load_model():
        sess = ort.InferenceSession("model/best.onnx", providers=["CPUExecutionProvider"])
        # Try to get class names from metadata if provided; otherwise create placeholders
        names = {}
        try:
            meta = sess.get_modelmeta().custom_metadata_map
            if "names" in meta:
                import json
                names = json.loads(meta["names"])
        except Exception:
            names = {}
        return {"sess": sess, "names": names}

    model_bundle = load_model()
    sess = model_bundle["sess"]
    class_names = model_bundle.get("names", {})

    # --- Preprocess (letterbox to maintain aspect ratio) ---
    img = pil_image.convert("RGB")
    orig_w, orig_h = img.size

    input_shape = sess.get_inputs()[0].shape  # e.g. [1,3,640,640] or [None,3,640,640]
    _, _, ih, iw = [s if s is not None else -1 for s in input_shape]
    if ih == -1 or iw == -1:
        # fallback default size
        ih, iw = 640, 640

    def letterbox(im, new_shape=(iw, ih), color=(114,114,114)):
        im_w, im_h = im.size
        r = min(new_shape[0] / im_w, new_shape[1] / im_h)
        new_unpad = (int(round(im_w * r)), int(round(im_h * r)))
        resized = im.resize(new_unpad, resample=Image.BILINEAR)
        new_im = Image.new('RGB', (new_shape[0], new_shape[1]), color)
        dw = (new_shape[0] - new_unpad[0]) // 2
        dh = (new_shape[1] - new_unpad[1]) // 2
        new_im.paste(resized, (dw, dh))
        return new_im, r, dw, dh

    img_resized, scale, pad_x, pad_y = letterbox(img, (iw, ih))
    x = np.array(img_resized).astype(np.float32) / 255.0
    # to CHW
    x = np.transpose(x, (2,0,1))[None, ...]

    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    outputs = sess.run(out_names, {input_name: x})
    # Many yolovX ONNX exports return a single output with shape (1, N, 85) or (1,25200,85)
    out = outputs[0]
    if out.ndim == 3:
        out = out[0]  # (N, 85)

    # --- Postprocess ---
    # Expected format: [x, y, w, h, conf, cls_probs...]
    detections = []
    if out.shape[1] >= 6:
        box_xywh = out[:, :4].copy()
        conf_obj = out[:, 4]
        class_scores = out[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_conf = class_scores[np.arange(len(class_ids)), class_ids]
        scores = conf_obj * class_conf

        # filter by confidence_threshold
        mask = scores >= confidence_threshold
        if mask.sum() == 0:
            return []

        box_xywh = box_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Convert from (x,y,w,h) centered on resized image to xyxy on original image
        xyxy = []
        for (x_c, y_c, w_box, h_box) in box_xywh:
            x1 = (x_c - w_box / 2 - pad_x) / scale
            y1 = (y_c - h_box / 2 - pad_y) / scale
            x2 = (x_c + w_box / 2 - pad_x) / scale
            y2 = (y_c + h_box / 2 - pad_y) / scale
            # clamp
            x1 = max(0, min(orig_w, x1)); y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2)); y2 = max(0, min(orig_h, y2))
            xyxy.append([x1, y1, x2, y2])

        xyxy = np.array(xyxy)

        # --- Simple NMS ---
        def nms(boxes, scores, iou_thresh=0.45):
            idxs = scores.argsort()[::-1]
            keep = []
            while idxs.size:
                i = idxs[0]
                keep.append(i)
                if idxs.size == 1:
                    break
                ious = []
                xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
                yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
                xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
                yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
                area_others = (boxes[idxs[1:],2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1])
                union = area_i + area_others - inter
                iou = inter / (union + 1e-6)
                idxs = idxs[1:][iou <= iou_thresh]
            return keep

        keep = nms(xyxy, scores)

        for k in keep:
            x1, y1, x2, y2 = xyxy[k]
            w_box = int(round(x2 - x1))
            h_box = int(round(y2 - y1))
            cls_id = int(class_ids[k])
            cls_name = str(class_names.get(str(cls_id), f"class_{cls_id}"))
            detections.append({
                "box": [int(round(x1)), int(round(y1)), w_box, h_box],
                "confidence": float(scores[k]),
                "class": cls_name
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