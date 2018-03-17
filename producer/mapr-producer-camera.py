from mapr_streams_python import Producer
import numpy as np
import cv2, time, pickle

p = Producer({'streams.producer.default.stream': '/mapr/DLcluster/tmp/rawcamerastream'})
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)

while (cap.isOpened):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ret, jpeg = cv2.imencode('.png', image)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    p.produce('t1', jpeg.tostring())
    # p.produce('t1', pickle.dumps(frame, protocol=0))
    time.sleep(0.35)

p.flush()
cap.release()
cv2.destroyAllWindows()
