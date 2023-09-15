import itertools
import os
import argparse
import jittor as jt
import jittor.nn as nn

jt.flags.use_cuda = 1
import numpy as np
from tqdm import tqdm
import jittor.transform as transforms
from PIL import Image
import src.resnet as resnet_model
from src.singlecropdataset import InferImageFolder
from src.utils import bool_flag
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--dump_path", type=str, default=None, help="The path to save results.")
    parser.add_argument("--data_path", type=str, default=None, help="The path to ImagenetS dataset.")
    parser.add_argument("--pretrained", type=str, default=None, help="The model checkpoint file.")
    parser.add_argument("-a", "--arch", metavar="ARCH", help="The model architecture.")
    parser.add_argument("-c", "--num-classes", default=50, type=int, help="The number of classes.")
    parser.add_argument("-t", "--threshold", default=0, type=float,
                        help="The threshold to filter the 'others' categroies.")
    parser.add_argument("--test", action='store_true',
                        help="whether to save the logit. Enabled when finding the best threshold.")
    parser.add_argument("--centroid", type=str, default=None, help="The centroids of clustering.")
    parser.add_argument("--checkpoint_key", type=str, default='state_dict', help="key of model in checkpoint")

    args = parser.parse_args()

    return args


def main_worker(args):
    centroids = np.load(args.centroid)
    centroids = jt.array(centroids)
    centroids = jt.normalize(centroids, dim=1, p=2)

    # build model
    if 'resnet' in args.arch:
        model = resnet_model.__dict__[args.arch](hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='pixelattn')
    else:
        raise NotImplementedError()

    checkpoint = jt.load(args.pretrained)[args.checkpoint_key]
    for k in list(checkpoint.keys()):
        if k.startswith('module.'):
            checkpoint[k[len('module.'):]] = checkpoint[k]
            del checkpoint[k]
            k = k[len('module.'):]
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format(args.pretrained))
    model.eval()

    # build dataset
    data_path = os.path.join(args.data_path, args.mode)
    normalize = transforms.ImageNormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = InferImageFolder(
        root=data_path,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        num_gpus=jt.world_size
    )
    dataloader = dataset.set_attrs(
        batch_size=jt.world_size, num_workers=4, drop_last=False, shuffle=False
    )
    # dataset2 = InferImageFolder(
    #     root=data_path + "_last",
    #     transform=transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]
    #     ),
    #     num_gpus=jt.world_size
    # )
    # dataloader2 = dataset2.set_attrs(
    #     batch_size=jt.world_size, num_workers=4, drop_last=False, shuffle=False
    # )
    dump_path = os.path.join(args.dump_path, args.mode)

    if not jt.in_mpi or (jt.in_mpi and jt.rank == 0):
        for cate in os.listdir(data_path):
            if not os.path.exists(os.path.join(dump_path, cate)):
                os.makedirs(os.path.join(dump_path, cate))

    # import train label
    # if args.mode == 'train':
    #     label_path = args.centroid[:-13] + "train_labeled.txt"
    # else:
    #     label_path = args.centroid[:-13] + "val_labeled.txt"
    # label_list = []
    # with open(label_path, 'r') as f:
    #     for line in f.readlines():
    #         tmp = line.split(' ')
    #         label_list.append((tmp[0], int(tmp[1].replace('\n', ''))))

    for index, (images, path, height, width, _) in enumerate(tqdm(dataloader)):
        path = path[0]  # data/ImageNetS/ImageNetS50/validation/n01443537/ILSVRC2012_val_00004677.JPEG
        cate = path.split("/")[-2]  # n01443537
        name = path.split("/")[-1].split(".")[0]  # ILSVRC2012_val_00004677
        # print(path)
        # print(label_list[index])
        # label = label_list[index][1] + 1 # first class is bg

        with jt.no_grad():
            h = height.item()
            w = width.item()

            out, mask = model(images, mode='inference_pixel_attention')

            mask = nn.upsample(mask, (h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

            out = jt.normalize(out, dim=1, p=2)
            B, C, H, W = out.shape
            out = out.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

            cosine = jt.matmul(out, centroids.t())
            cosine = cosine.view(1, H, W, args.num_classes).permute(0, 3, 1, 2)

            logit = mask
            prediction = jt.argmax(cosine, dim=1, keepdims=True)[0] + 1  # 0 is bg
            prediction = nn.interpolate(prediction.float(), (h, w), mode="nearest").squeeze(0).squeeze(0)

            prediction[logit < args.threshold] = 0

            # tmp = prediction.cpu().numpy()
            # print(label)
            # # print(tmp.shape)
            # print(Counter(tmp.reshape(-1).tolist()))

            res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
            res[:, :, 0] = prediction % 256
            res[:, :, 1] = prediction // 256

            res = res.cpu().numpy()
            logit = logit.cpu().numpy()

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(dump_path, cate, name + ".png"))
            if args.test:
                np.save(os.path.join(dump_path, cate, name + ".npy"), logit)

            jt.clean_graph()
            jt.sync_all()
            jt.gc()

    # for images, path, height, width, _ in tqdm(dataloader2):
    #     path = path[0]  # data/ImageNetS/ImageNetS50/validation/n01443537/ILSVRC2012_val_00004677.JPEG
    #     cate = path.split("/")[-2]  # n01443537
    #     name = path.split("/")[-1].split(".")[0]  # ILSVRC2012_val_00004677
    #
    #     with jt.no_grad():
    #         h = height.item()
    #         w = width.item()
    #
    #         out, mask = model(images, mode='inference_pixel_attention')
    #
    #         mask = nn.upsample(mask, (h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    #
    #         out = jt.normalize(out, dim=1, p=2)
    #         B, C, H, W = out.shape
    #         out = out.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
    #
    #         cosine = jt.matmul(out, centroids.t())
    #         cosine = cosine.view(1, H, W, args.num_classes).permute(0, 3, 1, 2)
    #
    #         logit = mask
    #         prediction = jt.argmax(cosine, dim=1, keepdims=True)[0] + 1
    #         prediction = nn.interpolate(prediction.float(), (h, w), mode="nearest").squeeze(0).squeeze(0)
    #
    #         prediction[logit < args.threshold] = 0
    #
    #         res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
    #         res[:, :, 0] = prediction % 256
    #         res[:, :, 1] = prediction // 256
    #
    #         res = res.cpu().numpy()
    #         logit = logit.cpu().numpy()
    #
    #         res = Image.fromarray(res.astype(np.uint8))
    #         res.save(os.path.join(dump_path, cate, name + ".png"))
    #         if args.test:
    #             np.save(os.path.join(dump_path, cate, name + ".npy"), logit)
    #
    #         jt.clean_graph()
    #         jt.sync_all()
    #         jt.gc()


if __name__ == "__main__":
    args = parse_args()
    main_worker(args=args)

    # print(label_list)
    # label_dic = {}
    # with open(label_path, 'r') as f:
    #     for line in f.readlines():
    #         tmp = line.split(' ')
    #         dir_name = tmp[0].split('/')[0]
    #         cate = tmp[1].replace('\n', '')
    #         if dir_name not in label_dic:
    #             label_dic[dir_name] = []
    #             label_dic[dir_name].append(cate)
    #         else:
    #             label_dic[dir_name].append(cate)

    # name = []
    # label = []
    # for k, v in label_dic.items():
    #     name.append(k)
    #     cnt = Counter(v)
    #     # print(cnt)
    #     # print(v.__len__())
    #     cate = cnt.most_common(2)
    #     label.append(cate)
    #
    # # for i in range(50):
    # #     print(name[i])
    # #     print(label[i])
    #
    # result = {}
    # done = [0]*50
    # flag = [0]*50
    # for j in range(50):
    #     if done[j] == 0:
    #         done[j] = 1
    #         # print(label[j][0][0])
    #         max_idx = j
    #         for k in range(j+1, 50):
    #             if done[k] == 1:
    #                 continue
    #             if label[j][0][0] == label[k][0][0]:
    #                 done[k] = 1
    #                 if label[k][0][1] > label[max_idx][0][1]:
    #                     max_idx = k
    #         result[name[max_idx]]=label[j][0][0]
    #         flag[max_idx]=1
    #         # print(max_idx)
    #         # print(flag)
    # print(result)
    #
    # s = set()
    # for k, v in result.items():
    #     s.add(v)
    # print(len(s))

    # name2 = []
    # label2 = []
    # for i in range(50):
    #     if flag[i] == 0:
    #         name2.append(name[i])
    #         label2.append(label[i])
    # print(name2)
    # print(label2)
    #
    # done = [0]*name2.__len__()
    # for j in range(name2.__len__()):
    #     if done[j] == 0:
    #         done[j] = 1
    #         print(label2[j][1][0])
    #         max_idx = j
    #         for k in range(j+1, name2.__len__()):
    #             if done[k] == 1:
    #                 continue
    #             if label2[j][1][0] == label2[k][1][0]:
    #                 done[k] = 1
    #                 if label2[k][1][1] > label2[max_idx][1][1]:
    #                     max_idx = k
    #         result[name2[max_idx]]=label2[j][1][0]

    # s = set()
    # for k, v in result.items():
    #     s.add(v)
    # print(len(s))
    #
    # del label_dic, label
