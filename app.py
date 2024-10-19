from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np

app = Flask(__name__)

# Make sure to create a folder named 'uploads' and 'outputs' in the same directory as this script
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the Haar Cascade classifier for face detection
face_cap = cv2.CascadeClassifier("C:/Users/Karan Vardhan Raj/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

# Function to apply deblurring (Wiener Filter approximation)
def deblur_image(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.float32) / np.prod(kernel_size)
    deblurred = cv2.filter2D(image, -1, kernel)
    return deblurred

# Function to resize the video resolution to custom width and height
def resize_image(image, new_width, new_height):
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized

# Function to enhance brightness and contrast
def enhance_brightness_contrast(image, alpha=2.0, beta=50):
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

# Function to enhance low light (Histogram Equalization)
def enhance_low_light(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return enhanced

# Function to process video
def process_video(input_path, output_path, custom_width, custom_height):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (custom_width, custom_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processing steps
        deblurred_frame = deblur_image(frame)
        bright_contrast_frame = enhance_brightness_contrast(deblurred_frame)
        enhanced_frame = enhance_low_light(bright_contrast_frame)
        resized_frame = resize_image(enhanced_frame, custom_width, custom_height)

        # Face detection
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(resized_frame)

    cap.release()
    out.release()

@app.route('/')
def home():
    return render_template('index.html')  # Ensure you have an index.html in the templates folder

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Allowed types: mp4, avi, mov'}), 400
    
    custom_width = int(request.form.get('custom_width', 720))  # Default width
    custom_height = int(request.form.get('custom_height', 480))  # Default height

    # Save the uploaded file
    input_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_file_path)

    # Output file path
    output_file_path = os.path.join(OUTPUT_FOLDER, f"enhanced_{file.filename}")

    try:
        # Process the video
        process_video(input_file_path, output_file_path, custom_width, custom_height)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return the output file
    return send_file(output_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
