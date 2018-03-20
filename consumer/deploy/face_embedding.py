from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

class FaceModel:
  def __init__(self, arggpu):
    model = edict()
    
    argthreshold = 1.24
    self.argflip = 0
    argdet = 2
    argmodel = '../models/model,0'
    argimage_size = '112,112'
    self.argdet = argdet 
    self.threshold = argthreshold
    self.det_minsize = 50
    self.det_threshold = [0.4,0.6,0.6]
    self.det_factor = 0.9
    _vec = argimage_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.image_size = image_size
    _vec = argmodel.split(',')
    assert len(_vec)==2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading',prefix, epoch)
    if arggpu >=0 :
      ctx = mx.gpu(arggpu)
    else:
      ctx = mx.cpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    #model.bind(data_shapes=[('data', (argbatch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (argbatch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  #def get_feature(self, face_img, bbox, points, img_orig, scale):
  def get_feature(self, face_img, bbox, points):
    #face_img is bgr image
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    face_img_1 = face_img.copy()
    #cv2.rectangle(img_orig, (int(round(bbox[0]/scale)), int(round(bbox[1]/scale))),
    #  (int(round(bbox[2]/scale)), int(round(bbox[3]/scale))),  (0, 255, 0), 2)
    cv2.rectangle(face_img_1, (int(round(bbox[0])), int(round(bbox[1]))),
      (int(round(bbox[2])), int(round(bbox[3]))),  (0, 255, 0), 2)
    aligned = np.transpose(nimg, (2,0,1))
    #print(nimg.shape)
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
        if self.argflip==0:
          break
        do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding, face_img_1 

  def get_initial_feature(self, face_img):
    #face_img is bgr image
    ret = self.detector.detect_face_limited(face_img, det_type = self.argdet)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    cv2.rectangle(face_img, (int(round(bbox[0])), int(round(bbox[1]))),
      (int(round(bbox[2])), int(round(bbox[3]))),  (0, 255, 0), 2)
    aligned = np.transpose(nimg, (2,0,1))
    #print(nimg.shape)
    embedding = None
    for flipid in [0,1]:
      if flipid==1:
        if self.argflip==0:
          break
        do_flip(aligned)
      input_blob = np.expand_dims(aligned, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      _embedding = self.model.get_outputs()[0].asnumpy()
      #print(_embedding.shape)
      if embedding is None:
        embedding = _embedding
      else:
        embedding += _embedding
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

