from ultralytics import settings
from ultralytics import YOLO
from roboflow import Roboflow

if __name__ == '__main__':

#TODO: MOVE TO SEPERATE FILE
#DOWNLOAD SMALL DATASET

    model = YOLO("yolo11m.pt")

    yamlFile = "c:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7_small\data.yaml"


    model.tune(data=yamlFile, epochs=50, iterations=60,  device='0', cache=True, batch=-1)
