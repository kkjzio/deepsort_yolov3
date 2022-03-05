import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .models import build_model
# from .train import input_size
from .model import Net

class Extractor(object):
    def __init__(self, model_name, model_path, use_cuda=True):
               
        self.device = "cuda" if torch.cuda.is_available(
        ) and use_cuda else "cpu"

        # [n,3,any(128),any(64)]-> [b,512]

        # use other featureNet
        self.net = build_model(name=model_name,
                num_classes=751,reid=True)  #osnet_small(96, reid=True) (751是用了Market1501数据集的关系)
        state_dict = torch.load(model_path)['net_dict']

        # state_dict.update({'classifier.weight':state_dict.pop('classifier.1.weight')})
        # state_dict.update({'classifier.bias':state_dict.pop('classifier.1.bias')})

        #originNet
        # self.net = Net(reid=True)
        # state_dict = torch.load(model_path)['net_dict']

        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.eval()
        self.net.to(self.device)
        self.size = (64,128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.3568, 0.3141, 0.2781],
                                 [0.1752, 0.1857, 0.1879])
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([
            self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops
        ],
                             dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("./244.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("mobilenetv2_x1_0","./checkpoint/ckpt.t7")
    feature = extr([img, img])
    print(feature.shape)
