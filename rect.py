import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import linear_model

import config_utils

recent_frame_thresh = config_utils.get('recent_frame_thresh')
frame_thresh = config_utils.get('frame_thresh')
dist_thresh = config_utils.get('dist_thresh')
max_samples_regress = config_utils.get('max_samples_regress')

class Rect:
    def __init__(self, x1, y1, x2, y2, score):
        self.x1 = x1        
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.xc = (x1 + x2)/2
        self.yc = (y1 + y2)/2
        self.w = x2 - x1
        self.h = y2 - y1
        self.score = score
    
    def getArea(self):
        return self.w * self.h

def getIOU(rect1, rect2):
    xleft = max(rect1.x1, rect2.x1)
    xright = min(rect1.x2, rect2.x2)
    ytop = max(rect1.y1, rect2.y1)
    ybottom = min(rect1.y2, rect2.y2)
    
    if xright < xleft or ybottom < ytop:
        return 0.0
        
    intersection_area = (xright - xleft) * (ybottom - ytop)
    return intersection_area / float(rect1.getArea() + rect2.getArea() - intersection_area)
           
def getDistance(rect1, rect2):
    return 1.0 - getIOU(rect1, rect2)

def matchRects(rects1, rects2):
    n = len(rects1)
    m = len(rects2)
    A = np.zeros((n, m))
    
    for i,j in np.ndindex(n, m):
        A[i, j] = getDistance(rects1[i], rects2[j])
        
    row_ind, col_ind = linear_sum_assignment(A)
    pairs = []
    
    for i,j in zip(row_ind, col_ind):
        if A[i, j] < dist_thresh:
            pairs.append((i,j))
            
    return pairs

def interpRect(rects):
    list_t = []
    list_rect = []
    
    n = min(frame_thresh, len(rects) + 1)
    
    for t in range(1, n):        
        if t == recent_frame_thresh and len(list_t) == 0:
            break
        
        rect = rects[-t]
        
        if rect != None:             
            list_t.append(t)
            list_rect.append(rect)
    
    if len(list_t) == 0:
        return None
    
    if len(list_t) == 1:
        return list_rect[0]
    
    if len(list_t) > max_samples_regress:
        list_t = list_t[:max_samples_regress]
        list_rect = list_rect[:max_samples_regress]
    
    list_t = [[t] for t in list_t]
    list_x1, list_y1, list_x2, list_y2 = [], [], [], []
    
    for rect in list_rect:
        list_x1.append(rect.x1)
        list_y1.append(rect.y1)
        list_x2.append(rect.x2)        
        list_y2.append(rect.y2)
    
    model = linear_model.LinearRegression()
    model.fit(list_t, list_x1)
    x1 = model.predict([[0]])[0]
    
    model = linear_model.LinearRegression()
    model.fit(list_t, list_y1)
    y1 = model.predict([[0]])[0]
    
    model = linear_model.LinearRegression()
    model.fit(list_t, list_x2)
    x2 = model.predict([[0]])[0]
    
    model = linear_model.LinearRegression()
    model.fit(list_t, list_y2)
    y2 = model.predict([[0]])[0]
    
    score = np.mean([rect.score for rect in list_rect])
    
    return Rect(x1, y1, x2, y2, score)