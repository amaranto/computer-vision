import glob
from ultralytics import YOLO
import torch

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("../runs/detect/train/weights/best.pt")
    #model = YOLO("yolov8m.yaml")
    results = model.train(
        data="./data.yaml",  
        save=True, 
        #workers=2, 
        resume=True,
        dropout=0.1,
        val=True,
        plots=True,
        device=device,
        imgsz = 640,
        save_period=10,
        # dfl = 4.5,
        # box = 7.5,
        cls = 0.7,
        epochs=50,  
        augment=True,
        perspective = 0.0,
        scale = 0.1,
        mixup = 0.0,
        mosaic = 0.0,
        shear = 0.0,
        degrees = 0.3,
        bgr = 0.0,
        copy_paste = 0.0
    )

    results = model.val()
    
    success = model.export(path="yolov8mTruco.pt")

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
        
