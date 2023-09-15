import argparse
import json
import os
import shutil

import numpy
import numpy as np
import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1
from PIL import Image
import jittor.transform as transforms
from tqdm import tqdm

import src.resnet as resnet_model
from src.singlecropdataset import InferImageFolder
from src.utils import hungarian
import colorsys
import random


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--dump_path',
                        type=str,
                        default=None,
                        help='The path to save results.')
    parser.add_argument('--match_file',
                        type=str,
                        default=None,
                        help='The matching file for test set.')
    parser.add_argument('--data_path',
                        type=str,
                        default=None,
                        help='The path to ImagenetS dataset.')
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help='The model checkpoint file.')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        help='The model architecture.')
    parser.add_argument('-c',
                        '--num-classes',
                        default=50,
                        type=int,
                        help='The number of classes.')
    parser.add_argument('--max_res', default=1000, type=int, help="Maximum resolution for evaluation. 0 for disable.")
    parser.add_argument('--method',
                        default='example submission',
                        help='Method name in method description file(.txt).')
    parser.add_argument('--train_data',
                        default='null',
                        help='Training data in method description file(.txt).')
    parser.add_argument(
        '--train_scheme',
        default='null',
        help='Training scheme in method description file(.txt), \
            e.g., SSL, Sup, SSL+Sup.')
    parser.add_argument(
        '--link',
        default='null',
        help='Paper/project link in method description file(.txt).')
    parser.add_argument(
        '--description',
        default='null',
        help='Method description in method description file(.txt).')
    args = parser.parse_args()

    return args


def main_worker(args):
    # build model
    if 'resnet' in args.arch:
        model = resnet_model.__dict__[args.arch](
            hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune', num_classes=args.num_classes)
    else:
        raise NotImplementedError()

    checkpoint = jt.load(args.pretrained)["state_dict"]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))
    model.eval()

    # build dataset
    assert args.mode in ['validation', 'test']
    data_path = os.path.join(args.data_path, args.mode)
    validation_segmentation = os.path.join(args.data_path,
                                           'validation-segmentation')
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    dataset = InferImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    dataloader = dataset.set_attrs(
        batch_size=1,
        num_workers=4)

    dump_path = os.path.join(args.dump_path, args.mode + '_mask')

    color_list = ncolors(50)
    color_list[0] = [168, 90, 240]
    for images, path, height, width, ori_image in tqdm(dataloader):
        path = path[0]
        cate = path.split('/')[-2]
        name = path.split('/')[-1].split('.')[0]
        if not os.path.exists(os.path.join(dump_path, cate)):
            os.makedirs(os.path.join(dump_path, cate))
        with jt.no_grad():
            H = height.item()
            W = width.item()

            output = model(images)

            if H * W > args.max_res * args.max_res and args.max_res > 0:
                output = nn.interpolate(output, (args.max_res, int(args.max_res * W / H)), mode="bilinear",
                                        align_corners=False)
                output = jt.argmax(output, dim=1, keepdims=True)[0]
                prediction = nn.interpolate(output.float(), (H, W), mode="nearest").long()
            else:
                output = nn.interpolate(output, (H, W), mode="bilinear", align_corners=False)
                prediction = jt.argmax(output, dim=1, keepdims=True)[0]

            prediction = prediction.squeeze(0).squeeze(0).numpy()
            color_list = numpy.array(color_list)
            # print(prediction.shape)
            # print(prediction)
            res = color_list[prediction]
            # print(res.shape)

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(dump_path, cate, name + '.png'))

            ori_image = ori_image.data.reshape((H, W, 3))
            ori_image = Image.fromarray(ori_image)
            # print(ori_image.shape)
            # ori_image.save(os.path.join(dump_path, cate, name + '_ori.png'))

            mask_img = Image.blend(ori_image, res, 0.5)
            # mask_img.save(os.path.join(dump_path, cate, name + '_mask.png'))


if __name__ == '__main__':
    args = parse_args()
    main_worker(args=args)
