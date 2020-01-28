import sys
import numpy as np
import cv2
import config_utils
from yolo import Model, array_to_image
from rect import matchRects, interpRect
import time

from threading import Thread
from queue import Queue

queue = Queue()
model_cfg = config_utils.get("model_cfg")
model_weights = config_utils.get("model_weights")
model_data = config_utils.get("model_data")
model = Model(model_cfg, model_weights, model_data)

width = config_utils.get("network_input_width")
height = config_utils.get("network_input_height")

ythresh = config_utils.get("counter_ythresh")
dy = config_utils.get("counter_dy")
max_history = config_utils.get("max_history")

upsampling = config_utils.get("upsampling") 
framerate = config_utils.get("video_framerate") / upsampling

color_list = [ '3366CC', 'DC3912', 'FF9900', '109618', '990099', '3B3EAC', '0099C6', 'DD4477', '66AA00' ,'B82E2E', '316395', '994499', '22AA99', 'AAAA11', '6633CC', 'E67300', '8B0707', '329262', '5574A6', '3B3EAC']

def getColor(code):
    r = int(code[:2], 16)
    g = int(code[2:4], 16)
    b = int(code[4:6], 16)
    return (r, g, b)
    
def getRects(img):    
    rects = model.detect(img)
    queue.put(rects)

def getNextFrame(cap):
    for _ in range(upsampling):
        ret, frame = cap.read()
        if frame is None:
            return None, None
            
    frame = cv2.resize(frame, (width, height))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = array_to_image(image)
    return frame, image

def getState(rect):
    if rect.yc <= ythresh - dy:
        return 'OUT'
        
    if rect.yc >= ythresh + dy:
        return 'IN'
    
    return 'UNK'

class Person:
    def __init__(self, id, init_rect):
        self.id = id
        self.route = [init_rect] 
        self.state = 'UNK'

if __name__ == '__main__':
    cap = cv2.VideoCapture(sys.argv[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')    
    out = cv2.VideoWriter('output.avi', fourcc, framerate, (width, height))
        
    persons = []
    max_id = 0
    num_in = num_out = 0
    i = 0
    logs = []

    start_time = time.time()
    frame = None
    next_frame, next_image = getNextFrame(cap)

    while next_frame is not None:
        t = i/framerate
        i += 1
        if i%100 == 0:
            print('Frame : ', i)
            
        image = next_image            
        thr = Thread(target=getRects, args=(image,))
        thr.start()
        
        if frame is not None:
            out.write(frame)
        
        frame = next_frame            
        next_frame, next_image = getNextFrame(cap)
        
        thr.join()
        rects = queue.get()
        
        if len(rects) > 0:        
            predict_rects = []  
            
            for person in persons:            
                predict_rect = interpRect(person.route)                
                if predict_rect != None:
                    predict_rect.person = person
                    predict_rects.append(predict_rect)
                    
            if len(predict_rects) > 0:
                pairs = matchRects(rects, predict_rects)
                
                for i1, i2 in pairs:
                    rects[i1].person = predict_rects[i2].person
            
            map_cur_rect = {}
            
            for rect in rects:
                if not hasattr(rect, 'person'):
                    max_id += 1
                    new_person = Person(max_id, rect)
                    rect.person = new_person
                    persons.append(new_person)
                
                map_cur_rect[rect.person.id] = rect
                
            for person in persons:
                cur_rect = map_cur_rect.get(person.id)
                person.route.append(cur_rect)
                person.route = person.route[-max_history:]
                                
                if cur_rect == None:
                    continue
                    
                cur_state = getState(cur_rect)
                    
                if person.state == 'OUT' and cur_state == 'IN':                    
                    logs.append({'time': t, 'direction': 'IN', 'xc': cur_rect.xc, 'yc': cur_rect.yc, 'pid': person.id})
                    num_in += 1

                if person.state == 'IN' and cur_state == 'OUT': 
                    logs.append({'time': t, 'direction': 'OUT', 'xc': cur_rect.xc, 'yc': cur_rect.yc, 'pid': person.id})
                    num_out += 1
                    
                if cur_state != 'UNK':
                    person.state = cur_state
                
            for rect in rects:
                color = getColor(color_list[rect.person.id % 20])
                cv2.rectangle(frame, (int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)), color , 1)
                cv2.putText(frame, str(rect.person.id), (int(rect.xc), int(rect.yc)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                                
        cv2.putText(frame, f'In : {num_in}', (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Out : {num_out}', (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (0, ythresh), (width, ythresh), (0, 255, 0), 1)
        
    out.release()
    cap.release()

    t = time.time() - start_time
    print('Time: ', t)
    
    with open('output.txt','w') as f:
        for log in logs:
            f.write(f"{log['time']}, {log['direction']}, {log['pid']}, {log['xc']}, {log['yc']}\n")