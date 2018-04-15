from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from symbol.resnet import *
from symbol.config import config
from symbol.processing import bbox_pred, clip_boxes, nms
import face_embedding
import numpy as np
import cv2, os, json, time
import mxnet as mx
from scipy import misc
import sys, argparse
import tensorflow as tf
import random, sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess


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

def getEmbedding(model, img_orig):
    ctx = mx.cpu()
    _, arg_params, aux_params = mx.model.load_checkpoint('mxnet-face-fr50', 0)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
    sym = resnet_50(num_class=2)

    img0, scale = resize(img_orig.copy(), 600, 1000)
    im_info = np.array([[img0.shape[0], img0.shape[1], scale]], dtype=np.float32)  # (h, w, scale)
    img = np.swapaxes(img0, 0, 2)
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
    keep = np.where(cls_scores >= 0.6)[0]
    cls_boxes = cls_boxes[keep, :]
    cls_scores = cls_scores[keep]
    dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets.astype(np.float32), 0.3)
    dets = dets[keep, :]
    print(dets.shape[0])
    i = 0
    bbox = dets[i, :4]
    roundfunc = lambda t: int(round(t/scale))
    vfunc = np.vectorize(roundfunc)
    bbox = vfunc(bbox)
    f2, jpeg = model.get_feature(img_orig, bbox, None)
    return f2

def getDistSim(args):
    model = face_embedding.FaceModel(args.gpuid)
    img = cv2.imread(args.image1)
    f1 = getEmbedding(model, img)
    img = cv2.imread(args.image2)
    f2 = getEmbedding(model, img)
    dist = np.sum(np.square(f1-f2))
    sim = np.dot(f1, f2.T)
    return dist, sim
    #diff = np.subtract(source_feature, target_feature)
    #dist = np.sum(np.square(diff),1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--gpuid', default=-1, type=int, help='')
    parser.add_argument('--image1', default='sam.jpg', type=str, help='')
    parser.add_argument('--image2', default='frances.jpg', type=str, help='')
    args = parser.parse_args()
    dist, sim = getDistSim(args)
    print('the similarity score is: {0:.4f}'.format(sim))
    print('the distance is: {0:.4f}'.format(dist))


