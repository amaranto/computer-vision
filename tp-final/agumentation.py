import glob, os
import cv2 as cv
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from config import BASE_PATH, yolo_ds_config, yolo_ds_dirs

OUTPUT_CHEK_FOLDER = os.getenv("OUTPUT_FOLDER", f"./{BASE_PATH}/augmentation-check/")
os.makedirs(OUTPUT_CHEK_FOLDER, exist_ok=True)


def draw_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2)->np.ndarray:

    x_min, y_min, w, h = bbox 
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    cv.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=color,
        lineType=cv.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name, show=False, output_file=None)->None:

    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = draw_bbox(img, bbox, class_name)

    if show:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    if output_file:
        augimg = Image.fromarray(img)
        augimg.save(output_file)

def agumentation(img, bboxes, augs, category_ids )->list[dict]:
    
    results = []

    for aug in augs:
        transform = aug
        
        
        transformed = transform(image=img, bboxes=bboxes.copy(), category_ids=category_ids)
        img_y, img_x,_ = transformed['image'].shape
        transformed_bboxes = [ [x-w/2,y-h/2,w,h] * np.array([img_x,img_y,img_x,img_y]) for x,y,w,h in transformed['bboxes'] ]
        transformed["bboxes_xywh"] = transformed_bboxes

        results.append(transformed)

    return results

def main():
    
    classes = { i: name for i,name in enumerate(yolo_ds_config["names"]) }

    for label_folder, image_folder in [ (yolo_ds_dirs["lbl_train"], yolo_ds_dirs["img_train"]) , ((yolo_ds_dirs["lbl_val"], yolo_ds_dirs["img_val"])) ]:
        images = glob.glob(image_folder + "/*.jpg")
        images += glob.glob(image_folder + "/*.png")
        images += glob.glob(image_folder + "/*.jpeg")
        images = images[:5]

        for img in images:

            img_base_name = os.path.basename(img).split(".")
            img_base_name = ".".join(img_base_name[:-1])

            orig = Image.open(img)
            image = np.asarray(orig)

            with open(f"{label_folder}/{img_base_name}.txt", "r") as f:
                content = f.readlines()
                b_boxes = [ [ float(p) for p in line.split()[1:] ] for line in content ]
                category_ids = [ int(line.split()[0]) for line in content ]

            aug_all = [
                A.Compose(
                    [
                        A.RandomBrightnessContrast(p=1.0),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                ),
                A.Compose(
                    [
                        A.RandomBrightnessContrast(p=1.0),
                        A.Perspective(p=1.0),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                ),                
                A.Compose(
                    [
                        A.HorizontalFlip(p=1.0),
                        A.RandomBrightnessContrast(p=1.0),
                        A.RandomRotate90(p=1.0),
                        A.MedianBlur(blur_limit=7, p=1.0),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                ),

                A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.MedianBlur(blur_limit=7, p=0.5)
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                ),
                A.Compose(
                    [
                        A.Resize(1024,1024),
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Flip(p=1.0),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                ),
                A.Compose(
                    [
                        A.Resize(1024,1024),
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomRotate90(p=0.5),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                ),                
                A.Compose(
                    [
                        A.Resize(640,640),
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomRotate90(p=0.5),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                ),      
                A.Compose(
                    [
                        A.MedianBlur(blur_limit=7, p=1.0),
                        A.RandomBrightnessContrast(p=0.2),
                        A.RandomRotate90(p=1.0),
                        A.Flip(p=1.0),
                    ],
                    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
                )            
            ]

            aug_imgs = agumentation(
                image, 
                b_boxes,
                aug_all,
                category_ids
            ) 

            for i,transformed in enumerate(aug_imgs):
                visualize(
                    transformed['image'],
                    transformed['bboxes_xywh'],
                    transformed['category_ids'],
                    classes,
                    output_file=f"{OUTPUT_CHEK_FOLDER}/{img_base_name}_{i}.png"
                )

                aug_img = Image.fromarray(transformed['image'])
                aug_img.save(f"{image_folder}/{img_base_name}_augmented_{i}.png")
                with open(f"{label_folder}/{img_base_name}_augmented_{i}.txt", "w") as f:
                    for i,(x,y,w,h) in enumerate(transformed['bboxes']):
                        f.write(f"{category_ids[i]} {x} {y} {w} {h}\n")   
main()