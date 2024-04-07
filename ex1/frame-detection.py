import cv2
import numpy as np

FILES = [
    "../media/tirada2.mp4",
]
for v in FILES:
    cap = cv2.VideoCapture(v)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    last_detected_area = set([])

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: 
            break

        # height, width, _ = frame.shape
        #frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
        #frame[:,:,0] = 0 
        #frame[:,:,1] = 0
        t,binary_frame = cv2.threshold(frame[:,:,2], 80, 255,cv2.THRESH_BINARY)
        #print(binary_frame)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)

        for i in range(1, n_labels):
            x1 = stats[i, cv2.CC_STAT_LEFT] 
            y1 = stats[i, cv2.CC_STAT_TOP] 
            w = stats[i, cv2.CC_STAT_WIDTH] 
            h = stats[i, cv2.CC_STAT_HEIGHT]        
            area = stats[i, cv2.CC_STAT_AREA]
            p1 = (x1,y1)
            p2 = (x1+w, y1+h)

            #proportion = ( p2[0] - p1[0])/ (p2[1]-p1[1])  
            #if proportion > 2 and  proportion <=2.8 :    
            if area > 3500 and area < 5000:
                if area in last_detected_area:
                    frame = cv2.rectangle( frame, p1, p2, (255,255,0), thickness=10)
                else:
                    last_detected_area.update([a for a in range(area-50,area+50)])
        cv2.imshow('Frame',frame)
        
        if cv2.waitKey(81) & 0xFF == ord('q'):
            break
        
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
