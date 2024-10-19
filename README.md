Project:
# Video Processing with OpenCV
This project uses OpenCV to enhance videos by applying deblurring, brightness and contrast adjustments, and face detection. The processed video can be saved to a specified output path with customizable resolution.

# Face_Enhancement.py 
contains the backend codes which enables to enhance the blurry images into enhanced with the help of opencv (machine Learning) to use deblur filiters and haarcascade (machine Learning technology) to detect the faces.

# frontend.html 
conatains the frontend code which is a space to upload video (media) for our main backend work.
# Tech Stack <br> opencv <br> HTMl <br> CSS <br> JavaScript <br> python <br> Flask

# Approach
The method utilizes **OpenCV** tools to improve and recreate portraitures from robbed quality CCTV footage. As a first step, the images are converted into grayscale to aid face detection with the use of **Haar Cascade Classifiers**. After a face has been located, blurring strategies such as **Gaussian** and **Median Blurs** are used in order to lessen the effects of noise, after which, edges are used to redefine such blurred facial regions. This approach simplifies the process of improving the quality of video footage making it easier to identify people in the said footage.

## Features
- **Deblurring**: Applies a Wiener Filter approximation to reduce blur in the video.
- **Brightness and Contrast Enhancement**: Enhances the brightness and contrast of each frame.
- **Low Light Enhancement**: Uses histogram equalization to improve visibility in low-light conditions.
- **Face Detection**: Detects faces in the video using a pre-trained Haar Cascade classifier.
- **Custom Resolution**: Allows the user to specify a custom output resolution for the processed video.

## Prerequisites

- Python
- OpenCV
- NumPy
- Flask

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/karan7955/Cipher_Sapiens1.git
   cd video-processing-opencv

2. **Install required packages:**:
   pip install opencv-contrib-python numpy


3. **Download Haar Cascade**:
   C:/Users/Karan Vardhan Raj/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml
   After Installing the opencv search the file **haarcascade_frontalface_default.xml** and copy the path location.

# Examples
user interference output


# Future Plans:
1. Integrate deep learning models like **SRGAN** for better face reconstruction.
2. Enable **real-time processing** for live CCTV feeds.
3. Implement **3D face reconstruction** for multi-angle footage.
4. Improve face recognition with advanced **AI integration**.
5. Optimize for **edge devices** to enhance performance in low-resource environments.
6. Incorporate **Explainable AI (XAI)** for transparency in model decisions.
7. Deploy the system on **cloud platforms** for scalability and accessibility.

## Acknowledgments
    **OpenCV** for the computer vision library.
    **NumPy** for numerical computations.
    **Flask** for building web applications.