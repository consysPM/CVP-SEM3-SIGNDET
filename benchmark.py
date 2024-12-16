from ultralytics.utils.benchmarks import benchmark

if __name__ == '__main__':
    yamlFile = "c:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7_small\\data.yaml"
    yoloFile = "runs-report/train_yolo11m_def_full_0_7/weights/epoch88.pt"
    # Benchmark on GPU
    benchmark(model=yoloFile, data=yamlFile, imgsz=640, half=False, device=0)