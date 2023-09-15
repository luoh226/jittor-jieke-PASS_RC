import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # validation_mask_sal_corrected
    folder_path = "./weights/res50w2/pixel_attention/0/validation/"
    file_names = os.listdir(folder_path)
    # print(file_names)
    for class_name in file_names:
        img_names = os.listdir(os.path.join(folder_path, class_name))
        save_path = os.path.join(folder_path[:-1] + '_mask_sal_corrected', class_name)
        # print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for img_name in img_names:
            if img_name[-3:] == 'npy':
                mask = np.load(os.path.join(folder_path, class_name, img_name))
                sal = np.load(os.path.join(folder_path[:-1] + '_sal', class_name, img_name))
                # print('='*50)
                # print(mask)
                # print(mask.shape)
                result = mask.copy()
                result[result < 0.49] = 0
                result[result >= 0.49] = 255
                result[sal < 0.014] = 0
                # result[sal >= 0.014] = 255
                # print(result)
                # print(result.shape)
                cv2.imwrite(os.path.join(save_path, img_name[:-3] + 'jpg'), result)

    # validation_sal_mask #
    # folder_path = "./weights/res50w2/pixel_attention/0/validation_sal/"
    # file_names = os.listdir(folder_path)
    # # print(file_names)
    # for class_name in file_names:
    #     img_names = os.listdir(os.path.join(folder_path, class_name))
    #     save_path = os.path.join(folder_path[:-1] + '_mask', class_name)
    #     # print(save_path)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     for img_name in img_names:
    #         if img_name[-3:] == 'npy':
    #             mask = np.load(os.path.join(folder_path, class_name, img_name))
    #             # print('='*50)
    #             # print(mask)
    #             # print(mask.shape)
    #             result = mask.copy()
    #             result[result < 0.014] = 0
    #             result[result >= 0.014] = 255
    #             # print(result)
    #             # print(result.shape)
    #             cv2.imwrite(os.path.join(save_path, img_name[:-3] + 'jpg'), result)

    # validation_gt_mask #
    # folder_path = "./weights/res50w2/pixel_attention/0/validation/"
    # file_names = os.listdir(folder_path)
    # # print(file_names)
    # for class_name in file_names:
    #     img_names = os.listdir(os.path.join(folder_path, class_name))
    #     save_path = os.path.join(folder_path[:-11], 'validation_gt_mask', class_name)
    #     # print(save_path)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     for img_name in img_names:
    #         if img_name[-3:] == 'npy':
    #             mask = Image.open(os.path.join('./data/ImageNetS/ImageNetS50/validation-segmentation', class_name, img_name[:-3] + 'png'))
    #             mask = np.array(mask)
    #             mask = mask[:, :, 1] * 256 + mask[:, :, 0]
    #             # print('='*50)
    #             # print(mask)
    #             # print(mask.shape)
    #             result = mask.copy()
    #             result[result != 0] = 255
    #             # print(result)
    #             # print(result.shape)
    #             cv2.imwrite(os.path.join(save_path, img_name[:-3] + 'jpg'), result)

    # validation_mask #
    # folder_path = "./weights/res50w2/pixel_attention/0/validation/"
    # file_names = os.listdir(folder_path)
    # # print(file_names)
    # for class_name in file_names:
    #     img_names = os.listdir(os.path.join(folder_path, class_name))
    #     save_path = os.path.join(folder_path[:-11], 'validation_mask', class_name)
    #     # print(save_path)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     for img_name in img_names:
    #         if img_name[-3:] == 'npy':
    #             mask = np.load(os.path.join(folder_path, class_name, img_name))
    #             # print('='*50)
    #             # print(mask)
    #             # print(mask.shape)
    #             result = mask.copy()
    #             result[result < 0.49] = 0
    #             result[result >= 0.49] = 255
    #             # print(result)
    #             # print(result.shape)
    #             cv2.imwrite(os.path.join(save_path, img_name[:-3] + 'jpg'), result)

    # Loss #
    # loss = []
    # with open('./weights/res101/slurm-16881.out') as f:
    #     for i, line in enumerate(f.readlines()):
    #         if 704 <= i <= 1080:
    #             line = line.strip('\n').split()
    #             if line[8][-5:] == '[800]':
    #                 loss.append(float(line[-4]))
    # x = np.array(range(1, len(loss) + 1))
    # plt.figure()
    # plt.plot(x, loss, label='ep20_lr6')
    # plt.title('swav loss when fine-tuning pixel attention head', fontsize=14)
    # plt.ylabel('Loss', fontsize=16)
    # plt.xlabel('epoch', fontsize=16)
    # xlabel = np.array(range(0, len(loss) + 1, 5))
    # plt.xticks(xlabel, xlabel)
    #
    # plt.legend()
    # plt.savefig('./weights/res101/pix_att_loss.jpg')
    #
    # loss2 = []
    # with open('./weights/res50w2/slurm-15205.out') as f:
    #     for i, line in enumerate(f.readlines()):
    #         if 1559 <= i <= 2094:
    #             # print(line)
    #             line = line.strip('\n').split()
    #             if line[8][-5:] == '[650]':
    #                 loss2.append(float(line[-4]))

    # x = np.array(range(1, len(loss2) + 1))
    # plt.plot(x, loss2, label='ep30_lr6')
    # xlabel = np.array(range(0, len(loss2) + 1, 5))
    # plt.xticks(xlabel, xlabel)
    # plt.legend()
    # plt.savefig('./weights/res50w2/pix_att_loss2.jpg')

    # mIoU #
    # loss2 = []
    # with open('./weights/res50w2/slurm-15205.out') as f:
    #     for i, line in enumerate(f.readlines()):
    #         if 1559 <= i <= 2094:
    #             # print(line)
    #             line = line.strip('\n').split()
    #             if line[8][-5:] == '[650]':
    #                 loss2.append(float(line[-4]))
    #
    # fig, ax = plt.subplots()
    # x = np.array(range(1, len(loss2) + 1))
    # line1, = ax.plot(x, loss2, color='#ff7f0e')
    # xlabel = np.array(range(0, len(loss2) + 1, 5))
    # ylabel = np.array(range(75, 251, 25))/100
    # ax.set_title('relationship between loss and mIoU, epoch=30, lr=6', fontsize=14)
    # ax.set_xticks(xlabel)
    # ax.set_yticks(ylabel)
    # ax.set_ylabel('Loss', fontsize=16)
    # ax.set_xlabel('epoch', fontsize=16)
    # ax.spines['right'].set_visible(False) # ax右轴隐藏
    #
    # miou = []
    # with open('./weights/res50w2/slurm-15255.out') as f:
    #     flag = 0
    #     for i, line in enumerate(f.readlines()):
    #         line = line.split()
    #         if flag == 1:
    #             # print(line)
    #             miou.append(float(line[1]))
    #             flag = 0
    #         if len(line) > 1 and line[1] == '0.80':
    #             flag = 1
    #
    # z_ax = ax.twinx() # 创建与轴群ax共享x轴的轴群z_ax
    # line2, = z_ax.plot(x, miou, color='#2ca02c')
    # z_ax.set_ylabel('mIoU', fontsize=16)
    # zlabel = np.array(range(15, 31, 2))
    # z_ax.set_yticks(zlabel)
    # plt.legend(handles=[line1, line2], labels=['loss', 'mIoU'])
    # plt.savefig('./weights/res50w2/pix_att_mIoU.jpg')