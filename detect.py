import cv2
import torch
import numpy as np

from yolo import Inference
from tools.tool import frames_crop, filter_too_small_single

class Running():
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.yolo = Inference()
        self.size = (1280, 960)

    def run(self):
        print('hhd')

        while True:
            frames = []
            # 开始堆叠视频帧
            try:
                for i in range(4):
                    ret, frame = self.cap.read()
                    frame = cv2.resize(frame, (1280, 960))
                    frames.append(frame)
            except:
                self.out.release()
                print(self.video_path, 'release')
                break

            mid_frame = frames[2]
            bboxes = self.yolo.predict_single(mid_frame)
            # 去除太小的识别框
            bboxes = filter_too_small_single(bboxes, thresh=32 * 64)

            print(bboxes)

            crops = []
            for bbox in bboxes:
                crop = frames_crop(frames, bbox, sz=(100, 200))
                crop = np.array(crop)
                crops.append(crop)

            crops = np.array(crops)  # crops.shape (2, 4, 64, 32, 3)  为裁剪后的片段 代表共有两个检测框，每个检测框有4帧
            # print(crops.shape)

            # 保存裁剪后的图像进行观察
            # for i in range(4):
                # cv2.imwrite("temp/test" + str(i) +".jpg", crops[0][i])

            import ipdb 
            ipdb.set_trace()

            print('testing')








if __name__ == "__main__":
    
    video_name = './data/test_video.avi'
    demo = Running(video_name)
    demo.run()



