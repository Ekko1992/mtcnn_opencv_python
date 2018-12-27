import cv2
from cv2 import dnn

from mtcnn import detect_face, drawBoxes, tic, toc

import sys


def main():
    minsize = 20

    caffe_model_path = "models"

    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    image_path = 'test.jpg'

    img = cv2.imread(image_path)



    
    PNet = dnn.readNetFromCaffe(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel")
    RNet = dnn.readNetFromCaffe(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel")
    ONet = dnn.readNetFromCaffe(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel")

    PNet.setPreferableBackend(dnn.DNN_BACKEND_HALIDE)
    RNet.setPreferableBackend(dnn.DNN_BACKEND_HALIDE)
    ONet.setPreferableBackend(dnn.DNN_BACKEND_HALIDE)

    for i in range(0,1000):
	    # check rgb position
	    tic()
	    #boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
	    boundingboxes, points = detect_face(img, minsize, PNet, RNet, ONet, threshold, False, factor)
	    toc()

    img = drawBoxes(img, boundingboxes)
    cv2.imshow('img', img)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()
