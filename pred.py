from ultralytics import YOLO
from PIL import Image


# Load a model
#model = YOLO("runs-report/train_yolo11m_ga_full_0_7/weights/best.pt")
model = YOLO("runs/detect/train5/weights/best.pt")  # pretrained YOLO11n model

imgdir = r"C:\dev\iu-study\trafficsigns\src\GTSRB-1\custom"
img = imgdir + r"\t1.jpeg"
#img = "C:/dev/iu-study/trafficsigns/src/GTSRB-1/test/images/00001_00003_00022_png.rf.1fa735b507baebedd0f3285398245a52.jpg"
# Run batched inference on a list of images
#results = model(["C:/dev/iu-study/trafficsigns/src/GTSRB-1/test/images/00001_00003_00022_png.rf.1fa735b507baebedd0f3285398245a52.jpg"],)  # return a list of Results objects
results = model.predict(img, conf=0.1, imgsz=640, device='0')
#result.show()

for result in results:
    # Plot results image
    result.show()