from mapr_streams_python import Consumer, KafkaError
from flask import Flask, Response
import cv2, os, json, time
import numpy as np
import argparse

os.environ['LD_LIBRARY_PATH'] = "$LD_LIBRARY_PATH:/opt/mapr/lib:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/"
app = Flask(__name__)

@app.route('/')

def index():
    # return a multipart response
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def kafkastream():
    c = Consumer({'group.id': args.groupid,
              'default.topic.config': {'auto.offset.reset': 'earliest', 'enable.auto.commit': 'false'}})
    # c.subscribe(['/user/mapr/nextgenDLapp/rawvideostream:topic1'])
    c.subscribe([args.stream+':'+args.topic])
    running = True
    while running:
        msg = c.poll(timeout=0.2)
        if msg is None: continue
        if not msg.error():
            nparr = np.fromstring(msg.value(), np.uint8)
            image = cv2.imdecode(nparr, 1)
            ret, jpeg = cv2.imencode('.png', image)
            bytecode = jpeg.tobytes()
            time.sleep(args.timeout)
            yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + bytecode + b'\r\n\r\n')

        elif msg.error().code() != KafkaError._PARTITION_EOF:
            print(msg.error())
            running = False
    c.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mapr consumer settings')
    # specify which stream, topic to read from, what is the consumer group and port to use.
    parser.add_argument('--groupid', default='dong00', help='')
    parser.add_argument('--stream', default='/tmp/personalstream', help='')
    parser.add_argument('--topic', default='all', help='')
    parser.add_argument('--timeout', default='0.035', type=float, help='')
    parser.add_argument('--port', default='5010', type=int, help='')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=True)
