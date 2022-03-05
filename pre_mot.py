import argparse
import os
import random
import time
from os.path import join

import cv2
import numpy as np
import torch

from deep_sort import DeepSort
from predict import InferYOLOv3
from utils.utils import xyxy2xywh
from utils.utils_sort import COLORS_10, draw_bboxes


def xyxy2tlwh(x):
    '''
    (top left x, top left y,width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


class DeepSortDetector(object):
    """[summary]
    Arguments:
        yolov3:
            cfg - yolov3.cfg
            weights - weights/best.pt
            data - coco.data
            conf_thres - 0.5
            nms_thres - 0.4
            img_size - 416
        deep sort:
            deep_checkpoint - "deep_sort/deep/checkpoint/ckpt.t7"
            max_dist - 0.2
        video_path - "./data/videosample/vidoe1.mp4"
        output_file - "./data/videoresults/video1.txt"
        display_width - 800
        display_height - 600
        save_path = "./video1_out.mp4"
    """

    def __init__(
            self,
            cfg,
            weights,
            video_path,
            deep_checkpoint="deep_sort/deep/checkpoint/ckpt.t7",
            data="dataset1.data",
            output_file=None,
            img_size=416,
            display=False,
            nms_thres=0.4,
            conf_thres=0.5,
            max_dist=0.2,
            display_width=800,
            display_height=600,
            save_path=None):
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.vidCap = cv2.VideoCapture()
        self.yolov3 = InferYOLOv3(cfg, img_size, weights, data, device,
                                  conf_thres, nms_thres)
        self.deepsort = DeepSort(deep_checkpoint,
                                 max_dist)
        self.display = display
        self.video_path = video_path
        self.output_file = output_file
        self.save_path = save_path

        if self.display:
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test", display_width, display_height)

    def __enter__(self):
        assert os.path.isfile(self.video_path), "Error: path error"
        self.vidCap.open(self.video_path)
        self.im_width = int(self.vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.save_path, fourcc, 20,
                                          (self.im_width, self.im_height))
        assert self.vidCap.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        frame_no = -1
        # skip_no = 2

        if self.output_file:
            f = open(output_file, "w")

        while self.vidCap.grab():
            frame_no += 1

            # skip frames every n frames
            # if frame_no % skip_no != 0:
            #     continue

            # start time
            total_begin = time.time()

            _, img = self.vidCap.retrieve()

            # yolov3部分
            yolo_begin = time.time()
            bbox_xyxy, cls_conf, cls_ids = self.yolov3.predict(img)
            # [x1,y1,x2,y2]
            yolo_end = time.time()

            # deepsort部分
            ds_begin = time.time()
            if bbox_xyxy is not None:
                bbox_cxcywh = xyxy2xywh(bbox_xyxy)

                outputs = self.deepsort.update(bbox_cxcywh, cls_conf, img)

                if len(outputs) > 0:
                    # [x1,y1,x2,y2] id
                    bbox_xyxy = outputs[:, :4]
                    ids = outputs[:, -1]
                    img = draw_bboxes(img, bbox_xyxy, ids)

                    # frame,id,tlwh,1,-1,-1,-1
                    if self.output_file:
                        bbox_tlwh = xyxy2xywh(bbox_xyxy)
                        for i in range(len(bbox_tlwh)):
                            write_line = "%d,%d,%d,%d,%d,%d,1,-1,-1,-1\n" % (
                                frame_no + 1, outputs[i, -1],
                                int(bbox_tlwh[i][0]), int(bbox_tlwh[i][1]),
                                int(bbox_tlwh[i][2]), int(bbox_tlwh[i][3]))
                            f.write(write_line)
            ds_end = time.time()

            total_end = time.time()

            if frame_no % 500 == 0:
                print("frame:%04d|det:%.4f|deep sort:%.4f|total:%.4f|det p:%.2f%%|fps:%.2f" % (frame_no,
                                                                                               (yolo_end - yolo_begin),
                                                                                               (ds_end - ds_begin),
                                                                                               (total_end - total_begin),
                                                                                               ((yolo_end - yolo_begin) * 100 / (
                                                                                                   total_end - total_begin)),
                                                                                               (1 / (total_end - total_begin))))

            if self.display is True:
                cv2.imshow("Test", img)
                cv2.waitKey(1)

            if self.save_path:
                self.output.write(img)

        if self.output_file:
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('parser')
    # 必须指定
    parser.add_argument("--video_root", type=str, default="./data/videosample")
    parser.add_argument("--cfg", type=str, default="cfg/mobile-yolo-cem.cfg")
    parser.add_argument("--data", type=str, default="data/dataset4.data")
    parser.add_argument("--weights",
                        type=str,
                        default="../YOLOv3-complete-pruning-master/weights/dataset4-mobile-yolo-cem/best.pt")
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument(
        "--deep_checkpoint",
        type=str,
        default="deep_sort/deep/checkpoint/mobilenetv2_x1_0_best.pt")

    # 超参数
    parser.add_argument("--conf_thres", type=float, default=0.5)
    parser.add_argument("--nms_thres", type=float, default=0.3)
    parser.add_argument("--max_dist", type=float, default=0.4)

    # 展示
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)

    args = parser.parse_args()

    for folder in os.listdir(args.video_root):
        video_path = join(args.video_root, folder + "/" + folder + ".mp4")
        output_file = join("./data/videoresult", folder + ".txt")
        save_path = join("./output", folder + ".avi")
        print("#"*30)
        print("#"*10, folder, "#"*10)
        print("#"*30)
        with DeepSortDetector(args.cfg, args.weights, video_path,
                              args.deep_checkpoint, args.data, output_file,
                              args.img_size, args.display, args.nms_thres,
                              args.conf_thres, args.max_dist,
                              args.display_width, args.display_height,
                              save_path) as det:
            det.detect()

        avi_name = os.path.basename(video_path).split(".")[0]
        # os.system("ffmpeg -y -i ./output/%s.avi -r 10 -b:a 32k ./output/%s.mp4" %
        #           (avi_name, avi_name))
