import cv2
import numpy as np

# Load the Haar Cascade classifier for face detection
face_cap = cv2.CascadeClassifier("C:/Users/Karan Vardhan Raj/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

# Function to apply deblurring (Wiener Filter approximation)
def deblur_image(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.float32) / np.prod(kernel_size)
    deblurred = cv2.filter2D(image, -1, kernel)
    return deblurred

# Function to enhance brightness and contrast
def enhance_brightness_contrast(image, alpha=1.4, beta=50):
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)  # alpha: contrast, beta: brightness
    return enhanced

# Function to upscale the video resolution
def upscale_image(image, scale_factor=2):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    upscale = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)  # Upscale using bicubic interpolation
    return upscale

# Function to enhance low light (Histogram Equalization)
def enhance_low_light(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return enhanced

# Main function to process the video
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width * 2, frame_height * 2))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply deblurring
        deblurred_frame = deblur_image(frame)
        
        # Enhance brightness and contrast
        bright_contrast_frame = enhance_brightness_contrast(deblurred_frame)
        
        # Enhance low light using histogram equalization
        enhanced_frame = enhance_low_light(bright_contrast_frame)
        
        # Upscale the resolution
        upscale_frame = upscale_image(enhanced_frame, scale_factor=2)

        # Prepare for face detection
        gray = cv2.cvtColor(upscale_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cap.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(upscale_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the processed frame with detected faces to output
        out.write(upscale_frame)

        # Display the processed frame for testing
        cv2.imshow('Enhanced Video', upscale_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Path to input video
input_video = 'C:/Users/Karan Vardhan Raj/Downloads/sample.mp4'

# Path to output video
output_video = 'C:/Users/Karan Vardhan Raj/Downloads/new2.mp4'

# Process video
process_video(input_video, output_video)