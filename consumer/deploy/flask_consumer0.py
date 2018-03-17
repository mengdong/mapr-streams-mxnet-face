from mapr_streams_python import Consumer, KafkaError
from flask import Flask, Response
import cv2, os, json, time
import numpy as np

app = Flask(__name__)

@app.route('/')

def index():
    # return a multipart response
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def kafkastream():
    c = Consumer({'group.id': 'dong01',
              'default.topic.config': {'auto.offset.reset': 'earliest', 'enable.auto.commit': 'false'}})
    # c.subscribe(['/user/mapr/nextgenDLapp/rawvideostream:topic1'])
    c.subscribe(['/tmp/personalstream:all'])
    running = True
    while running:
        msg = c.poll(timeout=0.2)
        if msg is None: continue
        if not msg.error():
            nparr = np.fromstring(msg.value(), np.uint8)
            image = cv2.imdecode(nparr, 1)
            ret, jpeg = cv2.imencode('.png', image)
            bytecode = jpeg.tobytes()
            time.sleep(0.041)
            yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + bytecode + b'\r\n\r\n')

        elif msg.error().code() != KafkaError._PARTITION_EOF:
            print(msg.error())
            running = False
    c.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
