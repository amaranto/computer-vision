import gdown
import glob, shutil, os
import yaml
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import shutil

datasets = [
    "https://drive.google.com/file/d/1u7eRhAuarRWJNNMTsdRIh18fBL3Lhlka/view?usp=sharing",
]

BASE_PATH = "dataset"
RAW_DATASET = f"{BASE_PATH}/dowloaded/"
DST_DATASET = f"{BASE_PATH}/yolo-format"
CONVERTER_OUTPUT = f"{BASE_PATH}/bbox_check_folder"

os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(RAW_DATASET, exist_ok=True)
os.makedirs(CONVERTER_OUTPUT + "/img", exist_ok=True)
os.makedirs(CONVERTER_OUTPUT + "/labels", exist_ok=True)

for d in datasets:
    d = d.replace("/view?usp=sharing", "")
    splitted_url = d.split("/")
    url, id, output = splitted_url[:-1], splitted_url[-1], splitted_url[-1].split("?")[0]
    
    if "folders" in url:
        output = f"{BASE_PATH}/{output}"
        gdown.download_folder(d, output=output, quiet=False)
    else:
        d = d.replace("file/d/", "uc?id=")
        output = f"{BASE_PATH}/{output}.zip"
        gdown.download(d, output=output, quiet=False, fuzzy=True)
        shutil.unpack_archive(output, f"{RAW_DATASET}/")

yolo_ds_dirs = {
    "img_train": DST_DATASET + "/images/train/",
    "img_val": DST_DATASET + "/images/val/",
    "lbl_train": DST_DATASET + "/labels/train/",
    "lbl_val": DST_DATASET + "/labels/val/"
}

yolo_ds_config = {
"train": "./images/train/",
"val": "./images/val/",
"nc": 51,
"names": [
    "1O", "1C", "1E", "1B", 
    "2O", "2C", "2E", "2B", 
    "3O", "3C", "3E", "3B", 
    "4O", "4C", "4E", "4B", 
    "5O", "5C", "5E", "5B", 
    "6O", "6C", "6E", "6B", 
    "7O", "7C", "7E", "7B", 
    "8O", "8C", "8E", "8B", 
    "9O", "9C", "9E", "9B", 
    "10O", "10C", "10E", "10B", 
    "11O", "11C", "11E", "11B", 
    "12O", "12C", "12E", "12B", 
    "J", "SKIP", "SSKIP"
]
}

def bnd_box_to_yolo_line(box,img_size):
        (x_min, y_min) = (box[0], box[1])
        (w, h) = (box[2], box[3])
        x_max = x_min+w
        y_max = x_max+h
        
        x_center = float((x_min + x_max)) / 2 / img_size[1]
        y_center = float((y_min + y_max)) / 2 / img_size[0]

        w = float((x_max - x_min)) / img_size[1]
        h = float((y_max - y_min)) / img_size[0]

        return np.float64(x_center), np.float64(y_center), np.float64(w), np.float64(h)

dirnames = os.listdir(RAW_DATASET)
print(f"Found { len(dirnames) } folders but working on {dirnames[0]}")

working_dir = f"{RAW_DATASET}/{dirnames[0]}"

for label_file in glob.glob(f"{working_dir}/*.txt"):

    base_name = os.path.basename(label_file).split(".")[0]
    image_path = [ glob.glob(f"{working_dir}/{base_name}.{ext}") for ext in ["jpg", "png", "jpeg"] ]
    image_path = [ img[0] for img in image_path if img ]
    image_path = image_path[0] if image_path else None

    if not image_path:
        print(f"Image not found for {label_file}")

        continue

    # Saca acentos de los nombres de los archivos
    cadena_sin_acentos = unidecode.unidecode(label_file)
    if cadena_sin_acentos != label_file:
        print(f"Renaming {label_file} to {cadena_sin_acentos}")
        os.rename(label_file, cadena_sin_acentos)

        img_sin_acentos = unidecode.unidecode(image_path)
        print(f"Renaming {image_path} to {img_sin_acentos}")
        os.rename(image_path, img_sin_acentos)
        label_file = cadena_sin_acentos
        image_path = img_sin_acentos

    print(f"Processing {image_path}...")
    image = cv.imread(image_path)
    image_h, image_w, _ = image.shape

    new_lines = []

    with open(label_file) as f:
        content = f.readlines()
        for line in content:
            line = line.split()
            if len(line) == 0:
                continue
                
            elif len(line) != 5:
                class_name = line[0]
                points = [ float(p) for p in line[1:]]
                points = np.array(points).reshape(-1,2).astype(np.float32)
                points = points * np.array([image_w, image_h])
                points = points.astype(np.int32)

                x,y,w,h = cv.boundingRect(points)

                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.imwrite(CONVERTER_OUTPUT + "/img/" + base_name + ".png", image)
                yolo_x, yolo_y, yolo_w, yolo_h = bnd_box_to_yolo_line([x,y,w,h], [image_h, image_w])

                new_lines.append(f"{class_name} {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")

    if new_lines:
        with open(CONVERTER_OUTPUT + "/labels/" + base_name + ".txt", "w") as f:
            f.writelines(new_lines)

new_labels = glob.glob(f"{CONVERTER_OUTPUT}/labels/*.txt") 
print( f"Copying {len(new_labels)} files to original dataset {working_dir}")
for f in new_labels:
    shutil.copy(f, working_dir)

print("Creating YOLO dataset...")

for k,d in yolo_ds_dirs.items():
    os.makedirs(d, exist_ok=True)

with open(f'{DST_DATASET}/data.yaml', 'w') as outfile:
    yaml.dump(yolo_ds_config, outfile, default_flow_style=False)

with open(f'{DST_DATASET}/classes.txt', 'w') as outfile:
    outfile.writelines("\n".join(yolo_ds_config["names"]))

label_files = glob.glob(f"{working_dir}/*.txt")

val_files = set( random.choices( label_files, k=int(len(label_files)*0.20) ) )
print( f"Moving {len(val_files)} files to validation set {yolo_ds_dirs['lbl_val']}")

for f in val_files:
    shutil.move(f, yolo_ds_dirs["lbl_val"])

train_files = glob.glob(f"{working_dir}/*.txt")
print( f"Moving {len(train_files)} files to training set {yolo_ds_dirs['lbl_train']}")
for f in train_files:
    shutil.move(f, yolo_ds_dirs["lbl_train"])

for f_name in  glob.glob(f"{yolo_ds_dirs['lbl_val']}/*.txt"):
    base_name = os.path.basename(f_name).split(".")[0]
    img_name = glob.glob(f'{working_dir}/{base_name}.*')
    if img_name:
        img_name = img_name[0]
        print(f"Moving {img_name} to {yolo_ds_dirs['img_val']}")
        shutil.move(img_name, yolo_ds_dirs["img_val"])
    else:
        continue

for f_name in glob.glob(f"{yolo_ds_dirs['lbl_train']}/*.txt"):
    base_name = os.path.basename(f_name).split(".")[0]
    img_name = glob.glob(f'{working_dir}/{base_name}.*')
    if img_name:
        img_name = img_name[0]
        print(f"Moving {img_name} to {yolo_ds_dirs['img_train']}")
        shutil.move(img_name, yolo_ds_dirs["img_train"])
    else:
        continue
