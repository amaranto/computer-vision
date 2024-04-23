import torch
import supervision as sv  
from ultralytics import YOLO  
import numpy as np  


SOURCE_VIDEO_PATH = "../media/video1.mp4"  
TARGET_VIDEO_PATH = "../media_output/video1_processed.mp4"  
MODEL_NAME = "yolov8x.pt"  

CLASSES = {"person": 0, "dog":16}
CLASSES_IDX = {v: k for k, v in CLASSES.items()}

model = YOLO(MODEL_NAME)  
model.predict(classes=[ CLASSES["person"], CLASSES["dog"]])
tracker = sv.ByteTrack()  


bounding_box_annotator = sv.BoundingBoxAnnotator()  
label_annotator = sv.LabelAnnotator()  


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame)[0]  

    detections = sv.Detections.from_ultralytics(results)  
    detections = tracker.update_with_detections(detections)          

    if detections.class_id is not None and detections.tracker_id is not None:
        unique, counts = np.unique(detections.class_id, return_counts=True)
        dict_class = {k: v for k, v in zip(counts, unique)}
        main_class = dict_class.get(max(counts), None)

        detections_tmp = {
            "xyxy": [],
            "class_id": [],
            "tracker_id": [],
            "confidence": [],
            "data": {},
            "mask": None
        }
        
        for xyxy, mask,confidence, class_id, tracker_id,data in detections:
            if class_id == main_class:
                
                detections_tmp["xyxy"].append(xyxy)
                detections_tmp["tracker_id"].append(tracker_id)
                detections_tmp["confidence"].append(confidence)
                detections_tmp["class_id"].append(class_id)

        detections.tracker_id = np.asarray(detections_tmp["tracker_id"])
        detections.class_id = np.asarray(detections_tmp["class_id"])
        detections.xyxy = np.asarray(detections_tmp["xyxy"], dtype=np.float32)
        detections.confidence = np.asarray(detections_tmp["confidence"], dtype=np.float32)   

        labels = [f"#{tracker_id} { CLASSES_IDX[main_class] }" for tracker_id in detections.tracker_id]  

        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame  

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,  
    target_path=TARGET_VIDEO_PATH,  
    callback=callback  
)