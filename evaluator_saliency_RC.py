import argparse
import os
from multiprocessing import Manager, Process

import numpy as np
from tqdm import tqdm

import jittor as jt
from PIL import Image
from src.singlecropdataset import EvalDataset
from src.utils import hungarian
from src.metric import intersectionAndUnionGPU, fscore, IoUDifferentSizeGPUWithBoundary
from collections import Counter


def get_dataset(args, mode, threshold=None, match=None):
    if mode == 'train':
        gt_path = os.path.join(args.data_path, f'{mode}')
    else:
        gt_path = os.path.join(args.data_path, f'{mode}-segmentation')
    predict_path = os.path.join(args.predict_path, mode)
    # print(gt_path)
    # print(predict_path)
    dataset = EvalDataset(predict_path, gt_path, threshold=threshold, match=match, sal=True, mode=mode, label_path = args.predict_path)
    dataset.set_attrs(
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
    return dataset


def match(args, mode, num_classes, threshold):
    loader = get_dataset(args, mode)
    if isinstance(threshold, float):
        threshold = [threshold]

    if mode == 'train':
        img_path = f'{mode}'
    else:
        img_path = f'{mode}-segmentation'
    for t in threshold:
        for cate in os.listdir(os.path.join(args.data_path, img_path)):
            if not os.path.exists(os.path.join(args.predict_path, mode + "_sal_" + str(t), cate)):
                os.makedirs(os.path.join(args.predict_path, mode + "_sal_" + str(t), cate))

    for _, _, predict, _, logit, path, sal, lab in tqdm(loader):
        cate, name = path[0].split("/")[-2:]

        predict = predict.numpy()
        # logit = logit.numpy()
        sal = sal.numpy()
        for t in threshold:
            predict_ = predict.copy()
            # print(Counter(predict_.reshape(-1).tolist()))
            # predict_[logit < t] = 0
            # print(Counter(predict_.reshape(-1).tolist()))
            predict_[sal < 0.014] = 0
            # predict_[sal > 0.97] = lab.numpy()
            # print(Counter(predict_.reshape(-1).tolist()))

            predict_ = predict_.squeeze()
            # print(predict_.shape)

            res_sal = jt.zeros((predict_.shape[0], predict_.shape[1], 3))
            res_sal[:, :, 0] = predict_ % 256
            res_sal[:, :, 1] = predict_ // 256
            res_sal = res_sal.cpu().numpy()

            res_sal = Image.fromarray(res_sal.astype(np.uint8))
            if mode == 'train':
                res_sal.save(os.path.join(args.predict_path, mode + "_sal_" + str(t), cate, name[:-4] + "png"))
            else:
                res_sal.save(os.path.join(args.predict_path, mode + "_sal_" + str(t), cate, name[:-4] + ".png"))


def evaludation(args, mode):
    if args.curve:
        thresholds = [
            threshold / 100.0 for threshold in range(args.min, args.max + 1, 2)
        ]
    else:
        thresholds = [args.t / 100.0]
    match(args, mode, args.num_classes, threshold=thresholds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_path',
                        default=None,
                        type=str,
                        help='The path to the predictions.')
    parser.add_argument('--data_path',
                        default=None,
                        type=str,
                        help='The path to ImagenetS dataset')
    parser.add_argument('--mode',
                        type=str,
                        default='validation',
                        choices=['validation', 'train'],
                        help='Evaluating on the validation or test set.')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--t',
                        default=0,
                        type=float,
                        help='The used threshold when curve is disabled.')
    parser.add_argument('--min',
                        default=0,
                        type=int,
                        help='The minimum threshold when curve is enabled.')
    parser.add_argument('--max',
                        default=60,
                        type=int,
                        help='The maximum threshold when curve is enabled.')
    parser.add_argument('-c',
                        '--num_classes',
                        type=int,
                        default=50,
                        help='The number of classes.')
    parser.add_argument('--curve',
                        action='store_true',
                        help='Whether to try different thresholds.')
    # parser.add_argument("--sal",
    #                     default=False,
    #                     type=bool,
    #                     help="saliency inference")

    args = parser.parse_args()

    evaludation(args, args.mode)
