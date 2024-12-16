from ultralytics import settings
from ultralytics import YOLO
from roboflow import Roboflow
from ultralytics.utils.torch_utils import strip_optimizer



if __name__ == '__main__':

#TODO: MOVE TO SEPERATE FILE
#DOWNLOAD SMALL DATASET

    #checkpoint files sind größer, siehe: https://github.com/ultralytics/ultralytics/issues/1736
    #The strip_optimizer() function updates the checkpoint dictionary value for model, replacing it with ema, and sets ema and optimizer keys to None, which will reduce the checkpoint size by 3/4.
    #strip_optimizer(r'C:\dev\iu-study\trafficsigns\src\runs\detect\train4\weights\epoch1.pt')

    model = YOLO("yolo11m.pt")
    yamlFile = "c:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7\data.yaml"


    model.train(data=yamlFile, epochs=100, imgsz=640, save_period = 1, device='0', workers = 1, batch=-1, multi_scale=True, scale=1)



    # model.train(data=yamlFile, epochs=20, imgsz=640, save_period = 1, device='0', workers = 1, batch=-1,
    #             lr0= 0.00763,
    #             lrf= 0.00992,
    #             momentum= 0.92249,
    #             weight_decay= 0.00062,
    #             warmup_epochs= 2.03059,
    #             warmup_momentum= 0.47448,
    #             box= 9.17312,
    #             cls= 0.53363,
    #             dfl= 1.06423,
    #             hsv_h= 0.02394,
    #             hsv_s= 0.49488,
    #             hsv_v= 0.34449,
    #             degrees= 0.0,
    #             translate= 0.13447,
    #             scale= 0.5081,
    #             shear= 0.0,
    #             perspective= 0.0,
    #             flipud= 0.0,
    #             fliplr= 0.31375,
    #             bgr= 0.0,
    #             mosaic= 0.69647,
    #             mixup= 0.0,
    #             copy_paste= 0.0)
