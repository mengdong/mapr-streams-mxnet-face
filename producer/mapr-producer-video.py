from mapr_streams_python import Producer
import numpy as np
import cv2,time

#p = Producer({'streams.producer.default.stream': '/mapr/cluster3/tmp/rawvideostream'})
p = Producer({'streams.producer.default.stream': '/mapr/DLcluster/tmp/rawvideostream'})
cap = cv2.VideoCapture('3billboards.mp4')

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
    p.produce('topic1', jpeg.tostring())

p.flush()
cap.release()
cv2.destroyAllWindows()
