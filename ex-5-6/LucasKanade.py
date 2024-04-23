import numpy as np
import cv2

MAX_POINTS = 1000
SENSITIVITY = 3
TRACK = False

cap = cv2.VideoCapture('../media/video1.mp4')
# parametros para detección de esquinas ShiTomasi
if (cap.isOpened()== False): 
    raise Exception("Error opening video stream or file")

feature_params = dict( maxCorners = MAX_POINTS,
                       qualityLevel = 0.2,
                       minDistance = 5,
                       )
 
# Parámetros para el flujo óptico de Lucas Kanade
lk_params = dict( winSize = (30,30),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
 
# Crea algunos colores aleatorios
color = np.random.randint(0,255,(MAX_POINTS,3))
# Toma el primer cuadro y encuentra esquinas en él
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
 
# Crear una máscara de imagen para dibujar
mask = np.zeros_like(old_frame)
 
while(True):
  ret,frame = cap.read()
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # calcula optical flow
  p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
 
  # Select good points
  good_new = p1[st==1]
  good_old = p0[st==1]
  #print( len(good_new))

  coords = [(new, old) for new, old in zip(good_new,good_old) if abs(new[0]-old[0]) > SENSITIVITY or abs(new[1]-old[1]) > SENSITIVITY ]

  for i,(new,old) in enumerate(coords):
    a,b = new.ravel()
    a,b = int(a),int(b)

    c,d = old.ravel()
    c,d = int(c),int(d)

    try:
      mask = mask if not TRACK else cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
      frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    except Exception as e:
      print(e)

  img = cv2.add(frame,mask)
 
  cv2.imshow('frame',img)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
 
  old_gray = frame_gray.copy()
  p0 = good_new.reshape(-1,1,2)
 
cv2.destroyAllWindows()
cap.release()