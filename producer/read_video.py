import numpy as np
import cv2
import datetime, time

cap = cv2.VideoCapture('sko.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ret, jpeg = cv2.imencode('.png', image)
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    bytecode = jpeg.tostring()
    nparr = np.fromstring(bytecode, np.uint8)
    image = cv2.imdecode(nparr, 1)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame',frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
