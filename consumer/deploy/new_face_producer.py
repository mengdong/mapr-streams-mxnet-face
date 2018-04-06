from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from symbol.resnet import *
from symbol.config import config
from symbol.processing import bbox_pred, clip_boxes, nms
import face_embedding
from mapr_streams_python import Consumer, KafkaError, Producer
import numpy as np
import cv2, os, json, time, sys, pickle
import mxnet as mx
import argparse, random, sklearn
import tensorflow as tf
from scipy import misc
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
import face_image, face_preprocess
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def index():
    return Response(kafkastream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs

def resize(im, target_size, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def get_face_embedding(filename, arg_params, aux_params, sym, model, ctx):
    img_orig = cv2.imread(filename)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img, scale = resize(img_orig.copy(), 600, 1000)
    im_info = np.array([[img.shape[0], img.shape[1], scale]], dtype=np.float32)  # (h, w, scale)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (c, h, w) order
    img = img[np.newaxis, :]  # extend to (n, c, h, w)
    arg_params["data"] = mx.nd.array(img, ctx)
    arg_params["im_info"] = mx.nd.array(im_info, ctx)
    exe = sym.bind(ctx, arg_params, args_grad=None, grad_req="null", aux_states=aux_params)

    exe.forward(is_train=False)
    output_dict = {name: nd for name, nd in zip(sym.list_outputs(), exe.outputs)}
    rois = output_dict['rpn_rois_output'].asnumpy()[:, 1:]  # first column is index
    scores = output_dict['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output_dict['bbox_pred_reshape_output'].asnumpy()[0]
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, (im_info[0][0], im_info[0][1]))
    cls_boxes = pred_boxes[:, 4:8]
    cls_scores = scores[:, 1]
    keep = np.where(cls_scores >0.6)[0]
    cls_boxes = cls_boxes[keep, :]
    cls_scores = cls_scores[keep]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets.astype(np.float32), 0.3)
    dets = dets[keep, :]
    bbox = dets[0, :4]
    roundfunc = lambda t: int(round(t/scale))
    vfunc = np.vectorize(roundfunc)
    bbox = vfunc(bbox)
    f_vector, jpeg = model.get_feature(img_orig, bbox, None)
    fT = f_vector.T
    return fT

def kafkastream():
    if args.gpuid >= 0:
        ctx = mx.gpu(args.gpuid)
    else:
        ctx = mx.cpu()
    _, arg_params, aux_params = mx.model.load_checkpoint('mxnet-face-fr50', 0)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    sym = resnet_50(num_class=2)
    model = face_embedding.FaceModel(args.gpuid)

    f1T = get_face_embedding(args.filename, arg_params, aux_params, sym, model, ctx)

    c = Consumer({'group.id': args.groupid,
              'default.topic.config': {'auto.offset.reset': 'earliest', 'enable.auto.commit': 'false'}})
    c.subscribe([args.readstream+':'+args.readtopic])
    running = True
    p = Producer({'streams.producer.default.stream': args.writestream})

    while running:
        msg = c.poll(timeout=0)
        if msg is None: continue
        if not msg.error():
            pickle_vector = pickle.loads(msg.value())
            nparr = np.fromstring(pickle_vector[0], np.uint8)
            img_orig = cv2.imdecode(nparr, 1)

            bbox_vector = pickle_vector[1]
            print(len(bbox_vector))
            embedding_vector = pickle_vector[2]
            if len(embedding_vector) > 0:
                sim_vector = [np.dot(f, f1T) for f in embedding_vector]
                idx = sim_vector.index(max(sim_vector))
                bbox = bbox_vector[idx]
                sim = sim_vector[idx]
                if sim > args.threshold:
                    img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img, (int(round(bbox[0])), int(round(bbox[1]))),
                        (int(round(bbox[2])), int(round(bbox[3]))),  (0, 255, 0), 2)
                    ret, jpeg = cv2.imencode('.png', img)
                    bytecode = jpeg.tobytes()
                    time.sleep(args.timeout)
                    yield (b'--frame\r\n'
                        b'Content-Type: image/png\r\n\r\n' + bytecode + b'\r\n\r\n')
                    if args.writetostream:
                        p.produce(args.writetopic, jpeg.tostring())
                        print(args.writetopic)
        elif msg.error().code() != KafkaError._PARTITION_EOF:
            print(msg.error())
            running = False

    c.close()
    p.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mapr consumer settings')
    parser.add_argument('--groupid', default='dong001', help='mapr consumer to read from')
    parser.add_argument('--gpuid', default='-1', type=int, help='')
    parser.add_argument('--port', default='5013', type=int, help='')
    parser.add_argument('--threshold', default='0.3', type=float, help='')
    parser.add_argument('--readstream', default='/tmp/processedvideostream', help='')
    parser.add_argument('--writestream', default='/tmp/identifiedstream', help='')
    parser.add_argument('--timeout', default='0.3', type=float, help='')
    parser.add_argument('--writetostream', default='0', type=int, help='')
    parser.add_argument('--writetopic', default='sam', help='topic to write to')
    parser.add_argument('--readtopic', default='topic1', help='topic to write to')
    parser.add_argument('--filename', default='sam_.jpg', help='')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=True)

