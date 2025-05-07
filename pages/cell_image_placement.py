import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import random
import time

# Function to check if a new placement would cause collision
def check_collision(mask, x, y, cell_mask, threshold=0.1):
    h, w = cell_mask.shape[:2]
    # Check if placement is within bounds
    if y < 0 or x < 0 or y + h > mask.shape[0] or x + w > mask.shape[1]:
        return True
    
    roi = mask[y:y+h, x:x+w]
    
    # Check if ROI size matches (handles edge cases)
    if roi.shape != cell_mask.shape:
        return True
    
    # Check overlap with existing cells
    overlap = np.sum(np.logical_and(roi > 0, cell_mask > 0))
    max_overlap = np.sum(cell_mask > 0) * threshold
    
    return overlap > max_overlap

# Function to create Gaussian alpha blending
def enhance_cell_image(cell_img):
    # Ensure the image is RGBA
    if cell_img.shape[2] == 3:
        cell_img_rgba = cv2.cvtColor(cell_img, cv2.COLOR_RGB2RGBA)
    else:
        cell_img_rgba = cell_img.copy()
    
    # Create alpha mask based on cell content
    mask = np.any(cell_img_rgba[:,:,:3] > 20, axis=2).astype(np.uint8) * 255
    
    # Create smooth alpha using Gaussian method
    h, w = cell_img_rgba.shape[:2]
    center_y, center_x = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    
    # Calculate distance from center 
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Find approximate radius of the cell (distance to furthest non-zero pixel)
    nonzero_y, nonzero_x = np.where(mask > 0)
    if len(nonzero_y) > 0:
        distances = np.sqrt((nonzero_y - center_y)**2 + (nonzero_x - center_x)**2)
        cell_radius = np.percentile(distances, 95)  # Use 95th percentile for robustness
    else:
        cell_radius = max(h, w) / 2
        
    # Create gradient alpha that falls off from center
    alpha = np.exp(-0.5 * (dist_from_center / (cell_radius/2))**2) 
    
    # Apply mask to ensure only cell pixels get alpha
    alpha = alpha * (mask > 0)
    
    # Scale to 0-255 range
    alpha = (alpha * 255).astype(np.uint8)
    
    # Apply the calculated alpha channel
    cell_img_rgba[:,:,3] = alpha
    
    return cell_img_rgba

# Function to apply alpha blending
def alpha_blend(source, target, position):
    """Apply alpha blending at the specified position"""
    x, y = position
    h, w = source.shape[:2]
    
    # Check if the region is within bounds
    if y < 0 or x < 0 or y + h > target.shape[0] or x + w > target.shape[1]:
        return target
    
    result = target.copy()
    
    # Get the region of interest
    roi = result[y:y+h, x:x+w]
    
    # Check if ROI and source have compatible shapes
    if roi.shape[:2] != source.shape[:2]:
        # Crop the source to fit
        source = source[:roi.shape[0], :roi.shape[1]]
    
    # Alpha blending formula: dst = alpha * src + (1 - alpha) * dst
    alpha = source[:,:,3:4] / 255.0 if source.shape[2] == 4 else np.ones((h, w, 1))
    result[y:y+h, x:x+w, :3] = (alpha * source[:,:,:3] + (1 - alpha) * roi[:,:,:3]).astype(np.uint8)
    
    # Update alpha channel if present
    if result.shape[2] == 4:
        result[y:y+h, x:x+w, 3] = np.maximum(result[y:y+h, x:x+w, 3], source[:,:,3])
    
    return result

# Function to add labels to image
def add_labels_to_image(image, cell_positions, cell_labels, font_scale=0.6, color=(255, 0, 0, 255)):
    """Add text labels to cell positions"""
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", int(12 * font_scale))
    except IOError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(12 * font_scale))
        except IOError:
            font = ImageFont.load_default()
    
    # Add each label
    for (x, y, w, h), label in zip(cell_positions, cell_labels):
        # Position the label near the cell (at the top)
        text_x = x
        text_y = max(0, y - 20)  # Position above the cell
        
        # Add a slight background for better readability
        text_width, text_height = draw.textsize(label, font=font)
        draw.rectangle(
            [(text_x-2, text_y-2), (text_x+text_width+2, text_y+text_height+2)],
            fill=(255, 255, 255, 180)  # Semi-transparent white
        )
        
        # Draw the text
        draw.text((text_x, text_y), label, fill=color[:4], font=font)
    
    return np.array(img_pil)

# Helper function to convert hex color to RGBA
def hex_to_rgba(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (255,)

# Streamlit App
def page():
    st.title("Mix image generation")
    st.write("Upload a plane image and multiple cell images with labels to place them without collisions")

    # Initialize session state for persistent storage
    if 'plane_file' not in st.session_state:
        st.session_state.plane_file = None
    if 'cell_data' not in st.session_state:
        st.session_state.cell_data = []
    if 'result_images' not in st.session_state:
        st.session_state.result_images = {'no_labels': None, 'with_labels': None}

    # File uploaders in sidebar
    with st.sidebar:
        st.header("Upload Images")
        # Plane image uploader
        plane_file = st.file_uploader("Upload plane image (background)", type=["jpg", "jpeg", "png"], key="plane_uploader")
        if plane_file is not None:
            st.session_state.plane_file = plane_file

        # Create containers for cell uploads with labels
        st.header("Cell Images")
        
        # Add cell button
        if st.button("+ Add Cell"):
            st.session_state.cell_data.append({
                "file": None,
                "label": ""
            })
            st.rerun()
        
        # Display upload fields for each cell
        for i in range(len(st.session_state.cell_data)):
            with st.container():
                st.subheader(f"Cell {i+1}")
                cell_file = st.file_uploader(f"Upload cell image {i+1}", key=f"cell_file_{i}")
                cell_label = st.text_input(f"Label for cell {i+1}", key=f"cell_label_{i}")
                
                # Update session state with current values
                if cell_file is not None:
                    st.session_state.cell_data[i]["file"] = cell_file
                st.session_state.cell_data[i]["label"] = cell_label
                
                if st.button(f"Remove Cell {i+1}", key=f"remove_{i}"):
                    st.session_state.cell_data.pop(i)
                    st.rerun()
        
        # Parameters
        st.header("Parameters")
        blur_radius = st.slider("Blur Radius", 5, 30, 15, help="Higher values create smoother transitions")
        max_attempts = st.slider("Max placement attempts per cell", 10, 100, 50)
        overlap_threshold = st.slider("Collision threshold", 0.01, 0.2, 0.05, help="Lower values mean less overlap")
        cell_scale = st.slider("Cell size scaling", 0.1, 5.0, 1.0)
        font_scale = st.slider("Label font size", 0.5, 2.0, 1.0, help="Adjust label text size")
        label_color = st.color_picker("Label color", "#FF0000")
        
        place_button = st.button("Place Cells on Plane")

    # Main area for displaying the result
    main_col1, main_col2 = st.columns([2, 1])

    # Show progress
    if st.session_state.plane_file and st.session_state.cell_data and place_button:
        # Check if we have valid cell files
        cell_files = [cell["file"] for cell in st.session_state.cell_data if cell["file"] is not None]
        cell_labels = [cell["label"] if cell["label"] else f"Cell {i+1}" 
                      for i, cell in enumerate(st.session_state.cell_data) if cell["file"] is not None]
        
        if cell_files:
            # Load the plane image with alpha channel
            plane_img = Image.open(st.session_state.plane_file).convert("RGBA")
            plane_array = np.array(plane_img)
            
            # Create copies for processing
            result_img_no_labels = plane_array.copy()
            result_img_with_labels = plane_array.copy()
            mask = np.zeros((plane_array.shape[0], plane_array.shape[1]), dtype=np.uint8)
            
            # Store cell positions for later label placement
            cell_positions = []
            
            # Display preview of cells
            with main_col2:
                st.subheader("Cell Images")
                for i, (cell_file, label) in enumerate(zip(cell_files, cell_labels)):
                    cell_img = Image.open(cell_file).convert("RGBA")
                    st.image(cell_img, caption=f"{label}", width=100)
            
            # Process each cell image
            with main_col1:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (cell_file, label) in enumerate(zip(cell_files, cell_labels)):
                    status_text.write(f"Processing {label} ({i+1}/{len(cell_files)})...")
                    
                    # Load cell image
                    cell_img = Image.open(cell_file).convert("RGBA")
                    cell_array = np.array(cell_img)
                    
                    # Resize cell if needed
                    h, w = cell_array.shape[:2]
                    new_h, new_w = int(h * cell_scale), int(w * cell_scale)
                    cell_array = cv2.resize(cell_array, (new_w, new_h))
                    
                    # Apply Gaussian alpha blending
                    cell_array = enhance_cell_image(cell_array)
                    if blur_radius > 0:
                        alpha = cell_array[:,:,3]
                        blurred_alpha = cv2.GaussianBlur(alpha, (blur_radius, blur_radius), 0)
                        cell_array[:,:,3] = blurred_alpha
                    
                    # Create binary mask for collision detection
                    cell_mask = (cell_array[:,:,3] > 30).astype(np.uint8) * 255
                    
                    # Try to find a non-colliding position
                    placed = False
                    for attempt in range(max_attempts):
                        x = random.randint(0, plane_array.shape[1] - new_w)
                        y = random.randint(0, plane_array.shape[0] - new_h)
                        
                        if not check_collision(mask, x, y, cell_mask, threshold=overlap_threshold):
                            mask[y:y+new_h, x:x+new_w] = np.maximum(mask[y:y+new_h, x:x+new_w], cell_mask)
                            cell_positions.append((x, y, new_w, new_h))
                            result_img_no_labels = alpha_blend(cell_array, result_img_no_labels, (x, y))
                            result_img_with_labels = alpha_blend(cell_array, result_img_with_labels, (x, y))
                            placed = True
                            break
                    
                    if not placed:
                        st.warning(f"Could not place {label} after {max_attempts} attempts.")
                    
                    progress_bar.progress((i + 1) / len(cell_files))
                    time.sleep(0.1)
                
                # Add labels to the labeled image
                label_color_rgba = hex_to_rgba(label_color)
                result_img_with_labels = add_labels_to_image(
                    result_img_with_labels, 
                    cell_positions, 
                    cell_labels, 
                    font_scale=font_scale,
                    color=label_color_rgba
                )
                
                status_text.write("Processing complete!")
                
                # Store results in session state
                st.session_state.result_images['no_labels'] = Image.fromarray(result_img_no_labels)
                st.session_state.result_images['with_labels'] = Image.fromarray(result_img_with_labels)

    # Display results if available
    if st.session_state.result_images['no_labels'] is not None:
        with main_col1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Result Without Labels")
                st.image(st.session_state.result_images['no_labels'])
                
                # Add download button
                buf1 = io.BytesIO()
                st.session_state.result_images['no_labels'].save(buf1, format="PNG")
                st.download_button(
                    label="Download Without Labels",
                    data=buf1.getvalue(),
                    file_name="cells_no_labels.png",
                    mime="image/png",
                    key="download_no_labels"
                )
            
            with col2:
                st.subheader("Result With Labels")
                st.image(st.session_state.result_images['with_labels'])
                
                # Add download button
                buf2 = io.BytesIO()
                st.session_state.result_images['with_labels'].save(buf2, format="PNG")
                st.download_button(
                    label="Download With Labels",
                    data=buf2.getvalue(),
                    file_name="cells_with_labels.png",
                    mime="image/png",
                    key="download_with_labels"
                )
    else:
        # Show instructions
        with main_col1:
            st.info("Upload a plane image and cell images with labels, then click 'Place Cells on Plane' to see the result.")
            
            st.subheader("Example of what to expect:")
            st.write("1. Upload a plane/background image")
            st.write("2. Upload multiple cell images and add labels for each")
            st.write("3. The app will place cells randomly avoiding collisions")
            st.write("4. Cells will have smooth Gaussian blending with the background")
            st.write("5. Two images will be generated - one with labels and one without")
        
        with main_col2:
            st.subheader("Tips for Best Results")
            st.write("- Use cell images with transparent backgrounds")
            st.write("- Adjust blur radius for smoother transitions")
            st.write("- Customize label appearance with font size and color options")

if __name__ == "__main__":
    page()