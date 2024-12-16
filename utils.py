from pathlib import Path
import gdown
import os
import zipfile


def download_extract(id, dir_name):
    gfile = 'https://drive.google.com/uc?id=' + id
    current_dir = os.path.dirname(__file__)
    file = Path(current_dir, dir_name, 'tmp.zip')
    file.parent.mkdir(exist_ok=True)
    
    output = file.__str__()
    gdown.download(gfile, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(file.parent.__str__())
        print("Datei erfolgreich extrahiert.")

    return file.parent.__str__()

def find_yaml(start_directory):
    for root, _, files in os.walk(start_directory):
        if 'data.yaml' in files:
            return os.path.join(root, 'data.yaml')
    return None
