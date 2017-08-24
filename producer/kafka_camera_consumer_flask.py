from mapr_streams_python import Consumer, KafkaError
from flask import Flask, Response
import numpy as np
import cv2, os, json, time
import mxnet as mx
app = Flask(__name__)

@app.route('/')

def index():
    # return a multipart response
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def kafkastream():
    c = Consumer({'group.id': 'consumer1',
              'default.topic.config': {'auto.offset.reset': 'earliest', 'enable.auto.commit': 'false'}})
    c.subscribe(['/tmp/rawvideostream:topic1'])
    running = True
    while running:
        msg = c.poll(timeout=1.0)
        if msg is None: continue
        if not msg.error():
            nparr = np.fromstring(msg.value(), np.uint8)
            image = cv2.imdecode(nparr, 1)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.png', frame)
            bytecode = jpeg.tobytes()
            yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + bytecode + b'\r\n\r\n')

        elif msg.error().code() != KafkaError._PARTITION_EOF:
            print(msg.error())
            running = False
    c.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
