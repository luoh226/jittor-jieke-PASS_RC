import os
import jittor as jt
import jittor.nn as nn
import numpy as np
from PIL import Image
import jittor.dataset as datasets
from src.utils import get_mask_of_class, mask_to_boundary


class EvalSALDataset(datasets.ImageFolder):
    """Dataset for the evaluation."""

    def __init__(self, prediction_root, target_root, match=None, threshold=None, mode='validation', label_path=None):
        super(EvalSALDataset, self).__init__(target_root)

        self.match = match
        self.threshold = threshold

        self.prediction_lst = []
        self.logit_lst = []
        self.sal_lst = []
        self.target_lst = []
        self.path_list = []
        self.label_list = []

        # import train label
        if mode == 'train':
            label_path = os.path.join(label_path, 'cluster', "train_labeled.txt")
        else:
            label_path = os.path.join(label_path, 'cluster', "val_labeled.txt")
        label_list_ = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                tmp = line.split(' ')
                # name = tmp[0].split('/')[1][:-5]
                cate = int(tmp[1].replace('\n', '')) + 1
                label_list_.append(cate)

        for index, (path, _) in enumerate(self.imgs):
            # print(path)
            categroy, name = path.split("/")[-2:]
            if mode == 'train':
                name = name[:-4] + 'png'
            # print(name)
            self.target_lst.append(path)
            self.prediction_lst.append(os.path.join(prediction_root, categroy, name))
            self.logit_lst.append(os.path.join(prediction_root, categroy, name[:-4] + ".npy"))
            self.sal_lst.append(os.path.join(prediction_root + '_sal', categroy, name[:-4] + ".npy"))
            self.path_list.append(path)
            self.label_list.append(label_list_[index])


    def _set_threshold_match(self, threshold, match):
        self.threshold = threshold
        self.match = match

    def __getitem__(self, item):
        """
        Returns:
        target (Tensor): Ground truth mask for semantic segmentation. (H x W)
        prediction (Tensor): Prediction mask. The value of each pixel indicates the assigned label. (H x W)
        logit (Tensor): Probility mask. The probability that each pixel is assigned the corresponding label. (H x W)
        """
        target = Image.open(self.target_lst[item])
        target = np.array(target)
        target = target[:, :, 1] * 256 + target[:, :, 0]

        prediction = Image.open(os.path.join(self.prediction_lst[item]))
        prediction = np.array(prediction)
        prediction = prediction[:, :, 1] * 256 + prediction[:, :, 0]

        # sal
        sal = 0
        if os.path.exists(os.path.join(self.sal_lst[item])):
            sal = np.load(os.path.join(self.sal_lst[item]), allow_pickle=True).astype(np.float32)

            if self.threshold is not None:
                pass
                prediction[sal < 0.014] = 0
                # prediction[sal > 0.98] = self.label_list[item]

        logit = 0
        if os.path.exists(os.path.join(self.logit_lst[item])):
            logit = np.load(os.path.join(self.logit_lst[item]), allow_pickle=True).astype(np.float32)

            if self.threshold is not None:
                # logit[sal < 0.014] = 0
                # logit[sal > 0.5] = self.threshold + 0.1

                prediction[logit < self.threshold] = 0
            logit = jt.array(logit)
            logit = logit.view(-1)

        if os.path.exists(os.path.join(self.sal_lst[item])):
            sal = jt.array(sal)
            sal = sal.view(-1)

        if self.match is not None:
            predict_matched = np.zeros_like(prediction)
            for k in np.unique(prediction):
                if k.item() == 0:
                    continue
                predict_matched[prediction == k.item()] = self.match[k.item() - 1] + 1
            prediction = predict_matched

        # Get boundary mask for each class.
        boundary_target = self.get_boundary_mask(target + 1)
        boundary_prediction = self.get_boundary_mask(prediction + 1)

        target = jt.float32(target.astype(np.float32))
        prediction = jt.array(prediction.astype(np.float32))
        boundary_target = jt.array(boundary_target.astype(np.float32))
        boundary_prediction = jt.array(boundary_prediction.astype(np.float32))


        target = target.view(-1)
        prediction = prediction.view(-1)
        boundary_target = boundary_target.view(-1)
        boundary_prediction = boundary_prediction.view(-1)

        mask = target != 1000
        target = target[mask]
        prediction = prediction[mask]
        boundary_target = boundary_target[mask]
        boundary_prediction = boundary_prediction[mask]
        if isinstance(logit, jt.Var):
            logit = logit[mask]
        if isinstance(sal, jt.Var):
            sal = sal[mask]
        return target, boundary_target, prediction, boundary_prediction, logit, sal, self.label_list[item]

    def __len__(self):
        return len(self.target_lst)

    def get_boundary_mask(self, mask):
        boundary = np.zeros_like(mask).astype(mask.dtype)
        for v in np.unique(mask):
            mask_v = get_mask_of_class(mask, v)
            boundary_v = mask_to_boundary(mask_v, dilation_ratio=0.03)
            boundary += (boundary_v > 0) * v
        return boundary


class EvalDataset(datasets.ImageFolder):
    """Dataset for the evaluation."""

    def __init__(self, prediction_root, target_root, match=None, threshold=None, sal=False, mode='validation', label_path=None):
        super(EvalDataset, self).__init__(target_root)

        self.match = match
        self.threshold = threshold
        self.sal = sal

        self.sal_lst = []
        self.prediction_lst = []
        self.logit_lst = []
        self.target_lst = []
        self.path_list = []
        self.label_list = []

        if self.sal:
            # import train label
            if mode == 'train':
                label_path = os.path.join(label_path, 'cluster', "train_labeled.txt")
            else:
                label_path = os.path.join(label_path, 'cluster', "val_labeled.txt")
            label_list_ = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    tmp = line.split(' ')
                    # name = tmp[0].split('/')[1][:-5]
                    cate = int(tmp[1].replace('\n', '')) + 1
                    label_list_.append(cate)

        for index, (path, _) in enumerate(self.imgs):
            categroy, name = path.split("/")[-2:]
            if mode == 'train':
                name = name[:-4] + 'png'
            # print(name)
            self.target_lst.append(path)
            self.prediction_lst.append(os.path.join(prediction_root, categroy, name))
            self.logit_lst.append(os.path.join(prediction_root, categroy, name[:-4] + ".npy"))
            self.sal_lst.append(os.path.join(prediction_root + '_sal', categroy, name[:-4] + ".npy"))
            self.path_list.append(path)
            if self.sal:
                self.label_list.append(label_list_[index])

    def _set_threshold_match(self, threshold, match):
        self.threshold = threshold
        self.match = match

    def __getitem__(self, item):
        """
        Returns:
        target (Tensor): Ground truth mask for semantic segmentation. (H x W)
        prediction (Tensor): Prediction mask. The value of each pixel indicates the assigned label. (H x W)
        logit (Tensor): Probility mask. The probability that each pixel is assigned the corresponding label. (H x W)
        """
        if not self.sal:
            target = Image.open(self.target_lst[item])
            target = np.array(target)
            target = target[:, :, 1] * 256 + target[:, :, 0]

        prediction = Image.open(os.path.join(self.prediction_lst[item]))
        prediction = np.array(prediction)
        prediction = prediction[:, :, 1] * 256 + prediction[:, :, 0]

        logit = 0
        if os.path.exists(os.path.join(self.logit_lst[item])):
            logit = np.load(os.path.join(self.logit_lst[item]), allow_pickle=True).astype(np.float32)

            if self.threshold is not None:
                prediction[logit < self.threshold] = 0
            logit = jt.array(logit)

            if not self.sal:
                logit = logit.view(-1)
        # sal
        sal = 0
        if os.path.exists(os.path.join(self.sal_lst[item])):
            sal = np.load(os.path.join(self.sal_lst[item]), allow_pickle=True).astype(np.float32)

            if self.threshold is not None:
                prediction[sal < 0.014] = 0
                # prediction[sal > 0.97] = self.label_list[item]
            sal = jt.array(sal)

            if not self.sal:
                sal = sal.view(-1)
        if self.sal:
            return 1, 1, prediction, 1, logit, self.path_list[item], sal, self.label_list[item]

        if self.match is not None:
            predict_matched = np.zeros_like(prediction)
            for k in np.unique(prediction):
                if k.item() == 0:
                    continue
                predict_matched[prediction == k.item()] = self.match[k.item() - 1] + 1
            prediction = predict_matched

        # Get boundary mask for each class.
        boundary_target = self.get_boundary_mask(target + 1)
        boundary_prediction = self.get_boundary_mask(prediction + 1)

        target = jt.float32(target.astype(np.float32))
        prediction = jt.array(prediction.astype(np.float32))
        boundary_target = jt.array(boundary_target.astype(np.float32))
        boundary_prediction = jt.array(boundary_prediction.astype(np.float32))


        target = target.view(-1)
        prediction = prediction.view(-1)
        boundary_target = boundary_target.view(-1)
        boundary_prediction = boundary_prediction.view(-1)

        mask = target != 1000
        target = target[mask]
        prediction = prediction[mask]
        boundary_target = boundary_target[mask]
        boundary_prediction = boundary_prediction[mask]
        if isinstance(logit, jt.Var):
            logit = logit[mask]

        return target, boundary_target, prediction, boundary_prediction, logit

    def __len__(self):
        return len(self.target_lst)

    def get_boundary_mask(self, mask):
        boundary = np.zeros_like(mask).astype(mask.dtype)
        for v in np.unique(mask):
            mask_v = get_mask_of_class(mask, v)
            boundary_v = mask_to_boundary(mask_v, dilation_ratio=0.03)
            boundary += (boundary_v > 0) * v
        return boundary


class ClusterImageFolder(datasets.ImageFolder):
    """Dataset for the clustering stage."""

    def __init__(self, root, transform):
        super(ClusterImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        """
        Returns:
        index (int): The index of an image in the dataset.
        path (str): The storage path of an image in the dataset.
        img (Tensor): The loaded images. (3 x H x W)
        label (int): The image-level ground truth label.
        taget (Tensor): The semantic segmentation label of an image.
        """
        tensors = super(ClusterImageFolder, self).__getitem__(index)
        img, label = tensors[:2]
        path = self.imgs[index][0]
        return index, path, img, label


class InferImageFolder(datasets.ImageFolder):
    """Dataset for the inference stage."""

    def __init__(self, root, transform, num_gpus=1):
        super().__init__(root, transform=transform)

        if len(self.imgs) % num_gpus != 0:
            padding = num_gpus - len(self.imgs) % num_gpus
            for i in range(padding):
                self.imgs.append(self.imgs[i])
            self.total_len = len(self.imgs)

    def __getitem__(self, index):
        """
        Returns:
        img (Tensor): The loaded images. (3 x H x W)
        path (str): The storage path of an image in the dataset.
        height (int): The height of an image.
        width (int): The width of an image.
        """
        path = self.imgs[index][0]
        img = Image.open(path).convert('RGB')
        ori_img = img
        height, width = img.size[1], img.size[0]

        if self.transform is not None:
            img = self.transform(img)
        return img, path, height, width, ori_img


class PseudoLabelDataset(datasets.ImageFolder):
    """Dataset for the finetuing stage."""

    def __init__(
            self,
            root,
            transform,
            pseudo_path,
            sal=False
    ):
        super(PseudoLabelDataset, self).__init__(root, transform)
        self.pseudo_path = pseudo_path
        self.sal = sal

    def __getitem__(self, index):
        """
        Returns:
        img (Tensor): The loaded image. (3 x H x W)
        pseudo (str): The generated pseudo label. (H x W)
        """
        path, _ = self.imgs[index]
        img = Image.open(path).convert('RGB')

        if self.sal:
            # print(path)
            pseudo = self.load_semantic(path, self.pseudo_path)
            pseudo = jt.array(np.array(pseudo)).permute(2, 0, 1).unsqueeze(0)
            pseudo = nn.interpolate(pseudo.float(), (img.size[1], img.size[0]), mode="nearest").squeeze(0)
            pseudo = Image.fromarray(pseudo.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            img, pseudo = self.transform(img, pseudo)

            return img, pseudo
        else:
            pseudo = self.load_semantic(path, self.pseudo_path)
            pseudo = jt.array(np.array(pseudo)).permute(2, 0, 1).unsqueeze(0)
            pseudo = nn.interpolate(pseudo.float(), (img.size[1], img.size[0]), mode="nearest").squeeze(0)
            pseudo = Image.fromarray(pseudo.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            img, pseudo = self.transform(img, pseudo)

            return img, pseudo

    def load_semantic(self, path, pseudo_path):
        cate, name = path.split('/')[-2:]
        name = name.replace("JPEG", "png")
        path = os.path.join(pseudo_path, cate, name)
        semantic = Image.open(path).convert('RGB')

        return semantic
