from ultralytics import settings
from ultralytics import YOLO
import sys
from utils import download_extract, find_yaml

if __name__ == '__main__':

    model = YOLO("yolo11m.pt")

    dir = download_extract("1cv6Sp9Tyukw61uAbWsPDGG1d0hm4_RLx", "small_dataset")
    yamlFile = find_yaml(dir)
    if(yamlFile == None):
        print("ERROR YAML FILE NOT FOUND")
        sys.exit(1)


    model.tune(data=yamlFile, epochs=50, iterations=60,  device='0', cache=True, batch=-1)
