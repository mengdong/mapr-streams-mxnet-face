import face_embedding
import argparse
import cv2
import numpy as np

def getDistSim(args):
    model = face_embedding.FaceModel(args.gpuid)
    img = cv2.imread(args.image1)
    f1 = model.get_initial_feature(img)
    img = cv2.imread(args.image2)
    f2 = model.get_initial_feature(img)
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

