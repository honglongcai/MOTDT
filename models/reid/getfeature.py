"""@Desc: ReID script version 2.1.0 for attention mgn model
@Result: Suning55ID: mAP: 90.1%
@Result: Suning763ID: mAP: 53.4%
@Time Consumption: 250fps under one GPU
@Date: created this script by Honglong Cai on 04/03/2019
@Data: updated by Honglong Cai on 04/22/2019
@Requirements:
    python >= 3.6
    pytorch >= 0.4.1
    torchvision >= 0.1.9
    PIL >= 5.3.0
@Image input: images can be inputted either as an single image \
        or from a directory
@CPU/GPU support: support both cpu and gpu
@Call format: python getfeature.py model_weight photo_path --sys_device_ids
        ----- photo_path can be either a single image or an image directory
        ----- use --sys_device_ids ''  when running at cpu environment
        ----- use --sys_device_ids 0  when running at gpu:0
        ----- use --sys_device_ids 0,1  or any when running at multiple gpu
@run at 'cpu': python getfeature.py /Users/honglongcai/model.pt \
               /Users/honglongcai/photo.jpg --sys_device_ids ''
               --sys_device_ids ''
@Test use 'gpu:0': python getfeature.py /Users/honglongcai/model.pt \
               /Users/honglongcai/photo.jpg --sys_device_ids 0
@testing code below shows the detail examples
"""


import argparse
import os

from PIL import Image
import numpy as np
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

from Model import Model
from data import Data


class GetFeature(object):
    """Extract features
    Arguments
        model_weight_file: pre-trained model
        sys_device_ids: cpu/gpu
    """
    def __init__(self, model_weight_file, sys_device_ids=''):
        if len(sys_device_ids) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = sys_device_ids
        self.sys_device_ids = sys_device_ids
        self.model = DataParallel(Model())
        if torch.cuda.is_available() and self.sys_device_ids != '':
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.model.load_state_dict(torch.load(model_weight_file,
                                              map_location=device))
        self.model.to(device)
        self.model.eval()
        
    def __call__(self, photo_path=None, batch_size=1):
        """
        get global feature and local feature
        :param photo_path : either photo directory or a single image
        :param batch_size : useful only when photo_path is a directory
        :return: feature: numpy array, dim = num_images * 2048,
                 photo_name: a list, len = num_images
        """
        '''
        if photo_dir is None and photo is None:
            raise self.InputError('Error: both photo_path '
                                  'and images is None.')
        if photo_dir and photo:
            raise self.InputError('Error: only need one argument, '
                                  'either photo_path or images.')
        '''
        # input is a directory
        if os.path.isdir(photo_path):
            dataset = Data(photo_path, self._img_process)
            data_loader = DataLoader(dataset, batch_size=batch_size,
                                     num_workers=8)
            features = torch.FloatTensor()
            photos = []
            for batch, (images, names) in enumerate(data_loader):
                images = images.float()
                if torch.cuda.is_available() and self.sys_device_ids != '':
                    images = images.to('cuda')
                feature = self.model(images).data.cpu()
                features = torch.cat((features, feature), 0)
                photos = photos + list(names)
                if batch % 10 == 0:
                    print('processing batch: {}'.format(batch))
            features = features.numpy()
            features = features/np.linalg.norm(features, axis=1,
                                               keepdims=True)
            return features, photos
        # input is a single image
        else:
            photo_name = photo_path.split('/')[-1]
            img = Image.open(photo_path)
            image = self._img_process(img)
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(image).float()
            feature = self.model(image).data.numpy()
            feature = feature/np.linalg.norm(feature, axis=1,
                                             keepdims=True)
            return feature, [photo_name]
    
    def _img_process(self, img):
        img = img.resize((128, 384), resample=3)
        img = np.asarray(img)
        img = img[:, :, :3]
        img = img.astype(float)
        img = img / 255
        im_mean = np.array([0.485, 0.456, 0.406])
        im_std = np.array([0.229, 0.224, 0.225])
        img = img - im_mean
        img = img / im_std
        img = np.transpose(img, (2, 0, 1))
        return img
        

if __name__ == '__main__':
    """testing code"""
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weight_file', type=str,
                        help='weight file')
    parser.add_argument('photo_path', type=str,
                        help='either a image directory or an image')
    parser.add_argument('--sys_device_ids', type=str, default='',
                        help='cuda ids')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    args = parser.parse_known_args()[0]
    
    get_feature = GetFeature(args.model_weight_file, args.sys_device_ids)
    features, p = get_feature(photo_path=args.photo_path, batch_size=args.batch_size)
    #print(features)
    #print(features.shape)
    #print(features.dtype)

