import numpy as np
import os
import cv2

def limit(x, min_, max_):
    return min(max_, max(x, min_))

def frames_crop(frames, bbox, sz=None):
    '''
        frames: list [[h,w,c], ...]
        bbox: [xmin, ymin, xmax, ymax]
        return:
            res: list [[h,w,c], ...], sub_window of frame
    '''
    h, w = frames[0].shape[:2]
    crops = []
    xmin, ymin, xmax, ymax = bbox
    xmin = limit(xmin, 0, w)
    ymin = limit(ymin, 0, h)
    xmax = limit(xmax, 0, w)
    ymax = limit(ymax, 0, h)

    for frame in frames:
        frame_crop = frame[ymin:ymax, xmin:xmax]
        if sz != None:
            frame_crop = cv2.resize(frame_crop, sz)
        crops.append(frame_crop)
    return crops

def area(bbox):
    xmin, ymin, xmax, ymax = bbox
    return (xmax-xmin) * (ymax-ymin)

def filter_too_small(bboxes, thresh):
    new_bboxes = [[] for _ in range(4)]
    for i in range(4):
        for j in range(len(bboxes[i])):
            if area(bboxes[i][j]) > thresh:
                new_bboxes[i].append(bboxes[i][j])

    return new_bboxes


def filter_too_small_single(bboxes, thresh):
    new_bboxes = []
    for bbox in bboxes:
        if area(bbox) > thresh:
            new_bboxes.append(bbox)

    return new_bboxes



