import torch
import time
import cv2
import numpy as np
import os
from PIL import Image

from models import *
from utils.datasets import *
from utils.utils import *

def totuple(a):
    '''
    list to tuple
    '''
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
# 指定cuda使用的设备编号
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def changtopic(im,out_img_size,bbox_x1y1x2y2, cls_conf, cls_ids, class_ind):
    '''
    get network output to marked picture by boxlist

    Parameters
    ----------
    im : `(h,w,c)` the format of *RGB*,oringin picture 
    out_img_size : `(h,w)` the shape of yoloV3's output
    bbox_x1y1x2y2 : `(ids,x1,y1,x2,y2)`
    cls_conf : `(ids,conf)`
    cls_ids : `(ids,class_num)`
    class_ind : the name of index list
    '''
    
    cmap = plt.get_cmap('rainbow')
    sp = im.shape
    colors = [cmap(i) for i in np.linspace(0, 1, len(cls_ids) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    for l in range(len(cls_ids)):
        if cls_conf[l] > 0.5:
            x1 = bbox_x1y1x2y2[l][0]
            y1 = bbox_x1y1x2y2[l][1]
            w = bbox_x1y1x2y2[l][2] - bbox_x1y1x2y2[l][0]
            h = bbox_x1y1x2y2[l][3] - bbox_x1y1x2y2[l][1]
            # print(x1,y1,w,h)
            
            # resize回原来的位置
            x1 = sp[1]*x1/out_img_size[1]
            y1 = sp[0]*y1/out_img_size[0]
            w = sp[1]*w/out_img_size[1]
            h = sp[0]*h/out_img_size[0]

            im = cv2.rectangle(im, np.uint16([x1,y1,w,h]),colors[l],2)

            # 做文字

            class_name = class_ind[cls_ids[l].astype(np.uint8)]
            # 取得文字的w，h
            retval, _=cv2.getTextSize(class_name, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
            im = cv2.rectangle(im, np.uint16([x1, y1-retval[1], retval[0], retval[1]]),colors[l],-1)
            im = cv2.putText(im, class_name, totuple(np.uint16([x1-1, y1-1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1 ,(255,255,255),1 )
    return im

class InferYOLOv3(object):
    def __init__(self,
                 cfg,
                 img_size,
                 weight_path,
                 data_cfg,
                 device,
                 conf_thres=0.5,
                 nms_thres=0.5):
        self.cfg = cfg
        self.img_size = img_size
        self.weight_path = weight_path
        # self.img_file = img_file
        self.device = device
        self.model = Darknet(cfg).to(device)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=device)['model'])
        self.model.to(device).eval()
        self.classes = load_classes(parse_data_cfg(data_cfg)['names'])
        self.colors = [random.randint(0, 255) for _ in range(3)]
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def predict(self, im0):
        # singleDataloader = LoadSingleImages(img_file, img_size=img_size)
        # path, img, im0 = singleDataloader.__next__()

        img, _, _ = letterbox(im0, new_shape=self.img_size)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0

        # TODO: how to get img and im0

        
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred, _ = self.model(img)
            # print(pred.shape)
            det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                      im0.shape).round()

            # Print results to screen
            # print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                # print('%g %ss' % (n, self.classes[int(c)]), end=', ')

            img = np.array(img.cpu())
            # Draw bounding boxes and labels of detections

            bboxes, confs, cls_confs, cls_ids = [], [], [], []
            dett = np.copy(det.cpu())
            for *xyxy, conf, cls_conf, cls_id in dett:
                # label = '%s %.2f' % (classes[int(cls_id)], conf)
                bboxes.append(xyxy)
                confs.append(conf)
                cls_confs.append(cls_conf)
                cls_ids.append(cls_id)
                # plot_one_box(xyxy, im0, label=label, color=colors)
            return bboxes, cls_confs, cls_ids
        else:
            return None, None, None

    def plot_bbox(self, ori_img, boxes):
        img = ori_img
        height, width = img.shape[:2]
        for box in boxes:
            # get x1 x2 x3 x4
            x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
            y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
            x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
            y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
            cls_conf = box[5]
            cls_id = box[6]
            # import random
            # color = random.choices(range(256),k=3)
            color = [int(x) for x in np.random.randint(256, size=3)]
            # put texts and rectangles
            img = cv2.putText(img, self.class_names[cls_id], (x1, y1),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        return img

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3,
                                     thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img,
                        label, (c1[0], c1[1] - 2),
                        0,
                        tl / 3, [225, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA)


if __name__ == "__main__":
    #################################################
    cfg = './cfg/yolov3-spp.cfg'
    # cfg = './cfg/yolov3.cfg'
    img_size = 416
    weight_path = './weights/yolov3-spp.pt'
    # weight_path = './weights/yolov3.pt'
    img_file = "./data/244.jpg"
    data_cfg = "./cfg/coco.data"
    conf_thres = 0.5
    nms_thres = 0.5
    # device = torch_utils.select_device()
    device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')

    #################################################
    yolo = InferYOLOv3(cfg, img_size, weight_path, data_cfg, device)
    # bbox_xcycwh, cls_conf, cls_ids = yolo(img_file)
    # print(bbox_xcycwh.shape, cls_conf.shape, cls_ids.shape)

    img = cv2.imread(img_file)
    # print(img)
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(img_size,img_size))
    # im = img
    print(im.shape)
    bbox_xcycwh, cls_conf, cls_ids = yolo.predict(im)
    class_ind=yolo.classes
    print(bbox_xcycwh.shape, cls_conf.shape, cls_ids.shape)
    # print(class_ind[2])
    # print(cls_ids)
    cls_c = 2
    bbox_xxyy =np.array([i for i,j in zip(bbox_xcycwh,cls_ids) if j == cls_c])
   

    result = changtopic(img,(img_size,img_size),bbox_xcycwh, cls_conf, cls_ids, class_ind)
    cv2.imwrite('result.png',result)
