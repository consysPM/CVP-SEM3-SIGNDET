import csv
from pathlib import Path
# importing shutil module
import shutil

def csv_to_yolo_labels(csv_file, source_folder, output_folder, type):
    """
    Convert a CSV file to YOLO format with individual text files for each image.

    Args:
        csv_file (str): Path to the input CSV file.
        images_folder (str): Path to the folder where images are stored.
        labels_folder (str): Path to the folder where labels will be saved.
    """

    new_labels_path = Path(output_folder, type, "labels")
    new_labels_path.mkdir(parents=True, exist_ok=True)  # Create the labels folder if it doesn't exist
    
    new_image_path = Path(output_folder, type, "images")
    new_image_path.mkdir(parents=True, exist_ok=True)  # Create the labels folder if it doesn't exist

    class_set = {0,1,2,3,4,5}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            width = int(row["Width"])
            height = int(row["Height"])
            x1 = int(row["Roi.X1"])
            y1 = int(row["Roi.Y1"])
            x2 = int(row["Roi.X2"])
            y2 = int(row["Roi.Y2"])
            class_id = int(row["ClassId"])
            image_path = row["Path"]


            if({class_id}.issubset(class_set) == False):
               continue

            # Convert to YOLO annotation format
            bbox_width = (x2 - x1) / width
            bbox_height = (y2 - y1) / height
            center_x = (x1 + x2) / 2 / width
            center_y = (y1 + y2) / 2 / height

            # Generate the corresponding label file name
            image_name = Path(image_path).stem  # Get the image file name without extension
            label_file = new_labels_path / f"{image_name}.txt"
            image_file = Path(source_folder, image_path)
            new_image_file = Path(new_image_path, Path(image_path).name);

            shutil.copyfile(image_file, new_image_file)

            # Write the YOLO annotation to the label file
            with open(label_file, 'a') as label_f:
                label_f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")



# Beispielaufruf für Trainingsdaten
csv_to_yolo_labels(
    csv_file="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\archive\\Train.csv", 
    source_folder="C:\\dev\iu-study\\trafficsigns\\GTSRB\\archive\\", 
    output_folder="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7\\",
    type="train"
)

csv_to_yolo_labels(
    csv_file="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\archive\\Test.csv", 
    source_folder="C:\\dev\iu-study\\trafficsigns\\GTSRB\\archive\\", 
    output_folder="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7\\",
    type="test"
)


# Beispielaufruf für Testdaten
# csv_to_yolo_yaml(
#     csv_file="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\archive\\Test.csv", 
#     yaml_file="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_full\\test_dataset.yaml", 
#     dataset_type="test"
# )
