import os 

BASE_PATH = os.getenv("DATA_PATH", "./datasets")
RAW_DATASET = f"{BASE_PATH}/dowloaded/"
YOLO_DATASET = f"{BASE_PATH}/yolo-format"
CONVERTER_OUTPUT = f"{BASE_PATH}/output/bbox_check_folder/"
SAVE_CHECK_IMAGE = os.getenv("SAVE_CHECK_IMAGE", "true").lower() == "true"

yolo_ds_dirs = {
    "img_train": YOLO_DATASET + "/images/train/",
    "img_val": YOLO_DATASET + "/images/val/",
    "lbl_train": YOLO_DATASET + "/labels/train/",
    "lbl_val": YOLO_DATASET + "/labels/val/"
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