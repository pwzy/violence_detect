
import sys
sys.path.insert(0, './yolov5')
import torch
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

IMAGE_SIZE = (1280, 960)

class Inference(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend('./checkpoints/yolov5s.pt', device=self.device)

        self.scale = transforms.Compose([transforms.Resize(size=IMAGE_SIZE[::-1])])

    def predict_single(self, image, conf_thres=0.45, iou_thres=0.45, max_det=1000, classes=0, agnostic_nms=False):
        '''
            pedestrian detection:
            images: numpy  shape: [h, w, 3]  rgb
            return [[xmin, ymin, xmax, xmax], ...]
        '''

        image = torch.tensor(image).permute(2,0,1).float()/255.0
        preds = self.model(image.to(self.device).unsqueeze(0))

        preds = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        preds = [[x[0],x[1],x[2],x[3]] for x in preds[:, :4].int().cpu().numpy()]

        return preds


if __name__ == "__main__":
    infer = Inference()
    image = cv2.imread('./yolov5/data/images/bus.jpg')

    image = cv2.resize(image, IMAGE_SIZE)

    bboxes = infer.predict_single(image)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 1)
    cv2.imshow('', image)
    cv2.waitKey(0)

    print(bboxes, image.shape)


