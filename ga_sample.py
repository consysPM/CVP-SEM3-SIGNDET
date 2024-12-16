import os
import random
from pathlib import Path
import shutil


def sample_annotations_by_class(annotations, num_samples=20):
    class_annotations = {}

    # Organize annotations by class
    for image_name, annots in annotations.items():
        for annot in annots:
            class_id = annot['class_id']
            if class_id not in class_annotations:
                class_annotations[class_id] = []
            class_annotations[class_id].append({**annot, "image_name": image_name})

    # Randomly sample annotations for each class
    sampled_annotations = {}
    for class_id, annots in class_annotations.items():
        sampled_annotations[class_id] = random.sample(annots, min(num_samples, len(annots)))

    return sampled_annotations

def read_yolo_labels(label_folder):

    annotations = {}

    # Iterate through all text files in the folder
    for filename in os.listdir(label_folder):
        if filename.endswith(".txt"):  # Process only .txt files
            filepath = os.path.join(label_folder, filename)
            image_name = os.path.splitext(filename)[0]  # Remove .txt extension
            annotations[image_name] = []

            # Read and parse the file
            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        
                        # Add the annotation to the list
                        annotations[image_name].append({
                            'class_id': class_id,
                            'center_x': center_x,
                            'center_y': center_y,
                            'bbox_width': bbox_width,
                            'bbox_height': bbox_height
                        })
    return annotations

def copy_files(source_dir, target_dir, anno):
    for class_id, samples in anno.items():
        for sample in samples:
             source_img = Path(source_dir, 'images', sample['image_name'] + '.png')
             target_img = Path(target_dir, 'images', sample['image_name'] + '.png')
             Path(target_img.parent).mkdir(parents=True, exist_ok=True)   

             source_label = Path(source_dir, 'labels', sample['image_name'] + '.txt')
             target_label = Path(target_dir, 'labels', sample['image_name'] + '.txt')
             Path(target_label.parent).mkdir(parents=True, exist_ok=True)   

             shutil.copyfile(source_img, target_img)
             shutil.copyfile(source_label, target_label)


        

# TRAIN
label_folder = "C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7\\train\\labels"  # Pfad zu Ihrem Label-Ordner
annotations = read_yolo_labels(label_folder)

sampled_annotations = sample_annotations_by_class(annotations, num_samples=30)

source_folder="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7\\train"
target_folder="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7_small\\train"

Path(target_folder).mkdir(parents=True, exist_ok=True)

copy_files(source_folder, target_folder, sampled_annotations)

#valid
# TRAIN
label_folder = "C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7\\valid\\labels"  # Pfad zu Ihrem Label-Ordner
annotations = read_yolo_labels(label_folder)

sampled_annotations = sample_annotations_by_class(annotations, num_samples=6)

source_folder="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7\\valid"
target_folder="C:\\dev\\iu-study\\trafficsigns\\GTSRB\\gtrsb_yaml_0_7_small\\valid"

Path(target_folder).mkdir(parents=True, exist_ok=True)

copy_files(source_folder, target_folder, sampled_annotations)