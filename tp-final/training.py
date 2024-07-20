import glob
from ultralytics import YOLO

if __name__=='__main__':
    
    #model = YOLO("../runs/detect/train3/weights/best.pt")
    model = YOLO("yolov8s.yaml").load("yolov8s.pt")
    results = model.train(
        data="./data.yaml",  
        save=True, 
        imgsz = 640,
        save_period=10,
        # dfl = 4.5,
        # box = 7.5,
        cls = 0.7,
        epochs=50,  
        workers=0, 
        device=0,
        augment=True,
        perspective = 0.0,
        scale = 0.1,
        mixup = 0.0,
        mosaic = 0.0,
        shear = 0.0,
        degrees = 0.0,
        bgr = 0.0,
        copy_paste = 0.0
    )

    results = model.val()
    
    #success = model.export(format="onnx")

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
        
