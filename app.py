from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import numpy as np
import io
import pickle
import base64
from skimage import measure
from skimage.morphology import closing, disk

app = Flask(__name__)

# Load the trained model
with open('digit_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_digit(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Resize while maintaining aspect ratio and pad to center in 28x28
    original_size = image.size
    aspect_ratio = original_size[0] / original_size[1]
    if aspect_ratio > 1:
        # Wider than tall
        new_width = 28
        new_height = int(28 / aspect_ratio)
    else:
        # Taller than wide
        new_height = 28
        new_width = int(28 * aspect_ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    # Create a new 28x28 image with black background
    new_image = Image.new('L', (28, 28), 0)
    # Paste the resized image centered
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    new_image.paste(image, (x_offset, y_offset))
    # Invert colors (MNIST digits are white on black)
    new_image = Image.eval(new_image, lambda x: 255 - x)
    # Convert to numpy array and flatten
    image_array = np.array(new_image).flatten() / 255.0
    # Predict
    prediction = model.predict([image_array])
    return int(prediction[0])

def predict_multi_digit(image):
    # Convert to grayscale
    image = image.convert('L')
    # Threshold to binary (adjust threshold if needed)
    image = image.point(lambda x: 0 if x < 128 else 255, 'L')
    # Convert to binary array
    binary_image = np.array(image) > 0
    # Use morphology to clean up
    binary_image = closing(binary_image, disk(1))
    # Label connected components
    labeled_image = measure.label(binary_image, connectivity=2)
    # Get bounding boxes
    regions = measure.regionprops(labeled_image)
    # Filter out small regions (noise)
    regions = [r for r in regions if r.area > 50]  # Adjust threshold as needed
    # Sort regions from left to right
    regions = sorted(regions, key=lambda r: r.bbox[1])
    # Predict each digit
    digits = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        # Create PIL image from binary region
        region_binary = binary_image[minr:maxr, minc:maxc]
        digit_image = Image.fromarray((region_binary * 255).astype(np.uint8), mode='L')
        if digit_image.size[0] > 0 and digit_image.size[1] > 0:
            digit_pred = predict_digit(digit_image)
            digits.append(str(digit_pred))
    # Combine digits into a number
    if digits:
        combined = int(''.join(digits))
        # Ensure it's within 0-99 (up to 100 as per user request, but 0-99 is 100 values)
        return min(combined, 99)
    else:
        return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return '<p>Kesalahan: Tidak ada bagian file</p>'
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return '<p>Kesalahan: Tidak ada file yang dipilih</p>'
    html_parts = []
    for file in files:
        if file.filename != '':
            try:
                # Read the image
                file_bytes = file.read()
                image = Image.open(io.BytesIO(file_bytes))
                # Ensure it's processed even if conversion fails
                predicted_digit = predict_multi_digit(image)
                # Encode image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_data = f"data:image/png;base64,{img_str}"
                html_parts.append(f'<div style="display: inline-block; text-align: center; margin: 10px;"><img src="{img_data}" style="max-width: 100px; margin: 10px;"><p style="color: black; font-weight: bold;">Prediksi: {predicted_digit}</p></div>')
            except Exception as e:
                html_parts.append('<div style="display: inline-block; text-align: center; margin: 10px;"><p>Tidak dapat diprediksi</p></div>')
        else:
            html_parts.append('<div style="display: inline-block; text-align: center; margin: 10px;"><p>File tidak valid</p></div>')
    return ''.join(html_parts)

if __name__ == '__main__':
    app.run(debug=True)
