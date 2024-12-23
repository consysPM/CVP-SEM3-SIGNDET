from ultralytics import YOLO
import cv2
from utils import download_extract


dir = download_extract("112TjqhFWlP5TO5iPLxxDvSxMJ15CKKCW", "yolo_model")
yolo_file = dir + "\\epoch69.pt"


# Load the YOLO model
#model = YOLO("runs-report/train_yolo11m_def_full_0_7_multiscale/weights/epoch69.pt")
model = YOLO(yolo_file)

# Open the video file
video_path = "test/vid.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame, conf=0.7)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()