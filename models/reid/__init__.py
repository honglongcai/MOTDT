import cv2
import numpy as np
from distutils.version import LooseVersion
import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel

from utils import bbox as bbox_utils
from utils.log import logger
from models import net_utils
from .Model import Model
from PIL import Image


def load_reid_model():
    model = DataParallel(Model())
    ckpt = '/home/honglongcai/Github/PretrainedModel/model_410.pt'
    model.load_state_dict(torch.load(ckpt, map_location='cuda'))
    logger.info('Load ReID model from {}'.format(ckpt))

    model = model.cuda()
    model.eval()
    return model


def img_process(img):
    img = np.asarray(img)
    img = Image.fromarray(img)
    img = img.resize((160, 384), resample=3)
    img = np.asarray(img)
    img = img[:, :, :3]
    img = img.astype(np.float32)
    img = img / 255
    im_mean = np.array([0.485, 0.456, 0.406])
    im_std = np.array([0.229, 0.224, 0.225])
    img = img - im_mean
    img = img / im_std
    img = np.transpose(img, (2, 0, 1))
    return img


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = bbox_utils.clip_boxes(bboxes, image.shape)
    patches = [img_process(image[box[1]:box[3], box[0]:box[2]]) for box in bboxes]
    return np.array(patches)


def extract_reid_features(reid_model, image, tlbrs):
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    patches = extract_image_patches(image, tlbrs)

    gpu = net_utils.get_device(reid_model)
    if LooseVersion(torch.__version__) > LooseVersion('0.3.1'):
        with torch.no_grad():
            im_var = Variable(torch.from_numpy(patches))
            if gpu is not None:
                im_var = im_var.cuda(gpu)
            features = reid_model(im_var).data
    else:
        im_var = Variable(torch.from_numpy(patches), volatile=True)
        if gpu is not None:
            im_var = im_var.cuda(gpu)
        features = reid_model(im_var).data

    return features
