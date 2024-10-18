import cv2

path = input("Enter the path of the video: ")
# path = "D:\IDM\Video\198887-909564506_small.mp4"
path = path.strip('"')

face_cap = cv2.CascadeClassifier("C:/Users/Karan Vardhan Raj/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture(path)
print("Cap = " , cap)


while True:
    ret , frame = cap.read()
    frame = cv2.resize(frame , (700 , 480))
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame , (x,y) , (x + h , y + h) , (0 , 255 , 0) , 2)
        cv2.imshow("Cap" , frame)
    # cv2.imshow("Gray" , gray)
    
    k = cv2.waitKey(25)
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()