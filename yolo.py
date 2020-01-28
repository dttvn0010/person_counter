from ctypes import *
import sys
import math
import random
import json
import os
import config_utils
from rect import Rect

darknet_libpath = config_utils.get('darknet_libpath')

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL(darknet_libpath, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)
    
def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr.astype('float32')/255.0).flatten()
    data = arr.ctypes.data_as(POINTER(c_float))
    return IMAGE(w,h,c,data)
    
class Model:
    def __init__(self, cfg_file, weight_file, meta_file, thresh=0.4, hier_thresh=0.4, nms=0.4, max_w_ratio=0.33):
        self.net = load_net(str.encode(cfg_file), str.encode(weight_file), 0)
        self.meta = load_meta(str.encode(meta_file))
        self.thresh = thresh
        self.hier_thresh = hier_thresh
        self.nms = nms
        self.max_w_ratio = max_w_ratio
            
    def detect(self, im):
        num = c_int(0)
        pnum = pointer(num)
        predict_image(self.net, im)
        dets = get_network_boxes(self.net, im.w, im.h, self.thresh, self.hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (self.nms): do_nms_obj(dets, num, self.meta.classes, self.nms);

        rects = []

        for j in range(num):
            if dets[j].prob[0] > 0:
                b = dets[j].bbox
                
                if b.w <= self.max_w_ratio * im.w:
                
                    x1 = int(b.x - b.w/2)
                    y1 = int(b.y - b.h/2)
                    x2 = int(b.x + b.w/2)
                    y2 = int(b.y + b.h/2)
                
                    rects.append(Rect(x1, y1, x2, y2, dets[j].prob[0]))

        free_detections(dets, num)
        return rects
        
import cv2
def test():
    model = Model("data/yolov3.cfg", "data/yolov3.weights", "data/coco.data")
    img = cv2.imread("sample/0001.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = model.detect(array_to_image(img))
    for rect in rects:
        print([rect.x1, rect.y1, rect.x2, rect.y2, rect.score])    
    
if __name__ == "__main__":
    test()