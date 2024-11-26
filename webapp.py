import io
from PIL import Image
import cv2
from flask import Flask, render_template, request, Response
import os
import time
from ultralytics import YOLO
import numpy as np
import base64

app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict_img", methods=["POST"])
def predict_img():
    if 'file' in request.files:
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        print(filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension in ['jpg', 'jpeg']:
            img = cv2.imread(filepath)
            frame = cv2.imencode(f'.{file_extension}', img)[1].tobytes()

            image = Image.open(io.BytesIO(frame))

            # Perform object detection
            yolo = YOLO('best_s.pt')
            results = yolo(image, save=False)
            res_plotted = results[0].plot()

            # Convert result to base64
            _, buffer = cv2.imencode(f'.{file_extension}', res_plotted)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('index.html', image_data1=image_base64)
    
    return "File format not supported or file not uploaded properly."

@app.route("/segment_plaque", methods=["POST"])
def segment_plaque():
    if 'file' in request.files:
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        file_extension = f.filename.rsplit('.', 1)[1].lower()

        if file_extension in ['jpg', 'jpeg']:
            # Read the image with OpenCV
            img = cv2.imread(filepath)
            
            # Convert BGR (OpenCV format) to RGB (PIL format)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            def segment_and_quantify_plaque_with_severity(image):
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

                # Split the LAB image into L, A, and B channels
                l, a, b = cv2.split(lab)

                # Threshold for plaque (yellow regions)
                plaque_mask = cv2.inRange(lab, np.array([160, 120, 150]), np.array([230, 140, 200]))

                # Refine the plaque mask
                kernel = np.ones((3, 3), np.uint8)
                plaque_mask = cv2.morphologyEx(plaque_mask, cv2.MORPH_CLOSE, kernel)

                # Define severity thresholds based on B channel values
                mild_threshold = 160
                moderate_threshold = 170
                severe_threshold = 190

                # Create empty masks for the different severities
                mild_mask = np.zeros_like(plaque_mask)
                moderate_mask = np.zeros_like(plaque_mask)
                severe_mask = np.zeros_like(plaque_mask)

                # Apply severity thresholds on the entire image where plaque is detected
                mild_mask[(plaque_mask > 0) & (b < mild_threshold)] = 255  # Mild
                moderate_mask[(plaque_mask > 0) & (b >= mild_threshold) & (b < moderate_threshold)] = 255  # Moderate
                severe_mask[(plaque_mask > 0) & (b >= moderate_threshold)] = 255  # Severe

                # Create color mappings for each severity
                output = image.copy()
                light_yellow = [255, 255, 0]    # Bright yellow for mild plaque
                darker_yellow = [210, 210, 0]   # Darker yellow for moderate plaque
                darkest_yellow = [100, 100, 0]  # Darkest yellow for severe plaque

                # Apply the colors to the respective severity regions
                output[mild_mask > 0] = light_yellow
                output[moderate_mask > 0] = darker_yellow
                output[severe_mask > 0] = darkest_yellow

                # Combine with the original image for visual comparison
                final_output = cv2.addWeighted(image, 0.7, output, 0.3, 0)

                # Calculate the severity percentages
                total_plaque_pixels = np.sum(plaque_mask > 0)
                mild_plaque = np.sum(mild_mask > 0)
                moderate_plaque = np.sum(moderate_mask > 0)
                severe_plaque = np.sum(severe_mask > 0)

                # Calculate percentages
                mild_percentage = (mild_plaque / total_plaque_pixels) * 100 if total_plaque_pixels > 0 else 0
                moderate_percentage = (moderate_plaque / total_plaque_pixels) * 100 if total_plaque_pixels > 0 else 0
                severe_percentage = (severe_plaque / total_plaque_pixels) * 100 if total_plaque_pixels > 0 else 0

                return final_output, mild_percentage, moderate_percentage, severe_percentage

            # Process the image
            results, mild, moderate, severe = segment_and_quantify_plaque_with_severity(img_rgb)

            # Convert the result image (RGB) back to BGR for OpenCV
            results_bgr = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)

            # Convert to base64
            _, buffer = cv2.imencode('.jpg', results_bgr)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Create data URL for HTML
            img_data = f"data:image/jpeg;base64,{img_base64}"

            return render_template('index.html', 
                                image_data=img_data, 
                                mild=round(mild, 2), 
                                moderate=round(moderate, 2), 
                                severe=round(severe, 2))

        return "File format not supported or file not uploaded properly."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)