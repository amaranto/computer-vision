import glob, os
import cv2 as cv
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./images")
BACKGROUND_FOLDER = os.getenv("BACKGROUND_IMAGE", "./background")
IMG_SIZE = (1024, 1024)

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/labels", exist_ok=True)
os.makedirs(f"{OUTPUT_FOLDER}/bboxes", exist_ok=True)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def bnd_box_to_yolo_line(box,img_size):
        (x_min, y_min) = (box[0], box[1])
        (w, h) = (box[2], box[3])
        x_max = x_min+w
        y_max = y_min+h
        
        x_center = float((x_min + x_max)) / 2 / img_size[1]
        y_center = float((y_min + y_max)) / 2 / img_size[0]

        w = float((x_max - x_min)) / img_size[1]
        h = float((y_max - y_min)) / img_size[0]

        return np.float64(x_center), np.float64(y_center), np.float64(w), np.float64(h)

def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):

    x_min, y_min, w, h = bbox 
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv.getTextSize(class_name, cv.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=color,
        lineType=cv.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name, show=False, output_file=None):

    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    if show:
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    if output_file:
        cv.imwrite(output_file, img)

def agumentation(img, bboxes, augs, img_base_name, category_ids=[0], classes={0: "313151"} ):
    
    results = []
    for i,aug in enumerate(augs):
        transform = aug

        img_x, img_y = img.size
        
        transformed = transform(image=np.asarray(img), bboxes=[ bnd_box_to_yolo_line([x,y,w,h],[ img_x, img_y ]) for _, x,y,w,h in bboxes], category_ids=category_ids)
        transformed_bboxes = [ [x-w/2,y-h/2,w,h] * np.array([img_x,img_y,img_x,img_y]) for x,y,w,h in transformed['bboxes'] ]
        transformed["bboxes_xywh"] = transformed_bboxes

        visualize(
            transformed['image'],
            transformed['bboxes_xywh'],
            transformed['category_ids'],
            classes,
            output_file=f"{OUTPUT_FOLDER}/bboxes/aug{i}_{img_base_name}"
        )

        cv.imwrite(f"{OUTPUT_FOLDER}/images/aug{i}_{img_base_name}", np.asarray(transformed['image']))
        with open(f"{OUTPUT_FOLDER}/labels/aug{i}_{img_base_name}.txt", "w") as f:
            for x,y,w,h in transformed['bboxes']:
                f.write(f"0 {x} {y} {w} {h}\n")    

        results.append(transformed)

    return results

def main():
    images = glob.glob("./images/313151/*.jpg")
    background_images = glob.glob(f"{BACKGROUND_FOLDER}/*.jpg")
    for img in images:

        img_base_name = os.path.basename(img)

        bg =  Image.open(background_images[ np.random.randint(0, len(background_images) - 1)]).resize(IMG_SIZE)
        orig = Image.open(img).resize(IMG_SIZE)
        image = np.asarray(orig)

        masks = mask_generator.generate(image)

       # ax = plt.gca()
        big_mask = masks[0]
        for mask in masks:
            if mask["area"] > big_mask["area"]:
                big_mask = mask
            #show_mask(mask["segmentation"], ax=ax, random_color=True)
        #plt.show()
        
        component_mask = big_mask["segmentation"].astype(np.uint8) * 255
        component_mask = ~component_mask

        n_labels, _, stats, _ = cv.connectedComponentsWithStats(component_mask, connectivity=4)
        b_boxes = [ (
                stats[i, cv.CC_STAT_AREA],
                stats[i, cv.CC_STAT_LEFT], 
                stats[i, cv.CC_STAT_TOP], 
                stats[i, cv.CC_STAT_WIDTH], 
                stats[i, cv.CC_STAT_HEIGHT]
            ) for i in range(1, n_labels) if stats[i, cv.CC_STAT_AREA] > 50000     
        ]

        pask_pil = Image.fromarray(big_mask["segmentation"])
        img_composed = Image.composite(bg, orig, pask_pil)  

        aug_all = [
            A.Compose(
                [
                    A.HorizontalFlip(p=1.0),
                    A.RandomBrightnessContrast(p=0.6),
                ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
            ),
            A.Compose(
                [
                    A.Resize(img_composed.size[0],img_composed.size[1]),
                    A.RandomBrightnessContrast(p=0.6),
                ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
            ),
            A.Compose(
                [
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.RandomBrightnessContrast(p=0.2),
                ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
            )            
        ]

        aug_1 = agumentation(
            img_composed, 
            b_boxes,
            aug_all,
            img_base_name
        ) 
        
main()