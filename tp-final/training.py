import glob
from ultralytics import YOLO
import torch

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = YOLO("../runs/detect/train/weights/best.pt")
    model = YOLO("yolov8m.yaml")
    results = model.train(
         data="./data.yaml",  
         save=True, 
         batch=0.8,
         cls=0.8,
         patience=20,
         #workers=2, 
         resume=False,
         dropout=0.1,
         val=True,
         plots=True,
         device=device,
         imgsz = 640,
         save_period=10,
         # dfl = 4.5,
         # box = 7.5,
         epochs=100,  
         hsv_s=0.7,
         hsv_v=0.4,
         hsv_h=0.2,
         augment=True,
         perspective = 0.001,
         scale = 0.1,
         mixup = 0.0,
         mosaic = 1.0,
         shear = 0.0,
         degrees = 0.3,
         bgr = 0.0,
         copy_paste = 1.0
     )

    results = model.val()
    
#    success = model.export()

    pictures = glob.glob("./datasets/custom/*")
    print (pictures)
    
    for picture in pictures:
        filename = picture.split("/")[-1]
        results = model(picture)  
        for result in results:
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            probs = result.probs
            obb = result.obb
            result.show()
            result.save(filename=f"./datasets/results/{filename}")
        
