import os
import cv2
from PIL import Image


def avg_cut_four_square(square_img_path, square_img_name, save_path=''):
    img = Image.open(square_img_path)
    size = img.size
    weight = int(size[0] // 2)
    height = int(size[1] // 2)
    now_iter_id = 1
    for i in range(2):
        for j in range(2):
            box = (weight * j, height * i, weight * (j + 1), height * (i + 1))
            region = img.crop(box)
            region.save(os.path.join(save_path, 'part{}'.format(now_iter_id) + square_img_name))
            now_iter_id += 1


def avg_cut_four_square_main(im_path,im_list, suffix,save_path=''):
    for im_name in im_list:
        img_path = os.path.join(im_path, "{0}_sat.{1}").format(im_name,suffix)
        avg_cut_four_square(img_path, "{0}_sat.{1}".format(im_name,suffix), save_path=save_path)
def avg_cut_four_square_main_gps(im_path,im_list, suffix,save_path=''):
    for im_name in im_list:
        img_path = os.path.join(im_path, "{0}_gps.{1}").format(im_name,suffix)
        avg_cut_four_square(img_path, "{0}_gps.{1}".format(im_name,suffix), save_path=save_path)
def avg_cut_four_square_main_mask(im_path,im_list, suffix,save_path=''):
    for im_name in im_list:
        img_path = os.path.join(im_path, "{0}_mask.{1}").format(im_name,suffix)
        avg_cut_four_square(img_path, "{0}_mask.{1}".format(im_name,suffix), save_path=save_path)

# how to use?
if __name__ == '__main__':
    image_list = [x[:-9] for x in os.listdir('E:/ML_data/remote_data/BJRoad/train_val/mask') if x.find('mask.png') != -1]

    ori_path = 'E:/ML_data/remote_data/BJRoad/train_val/'
    # 将ori_img_path目录下所有图片切割，且保存在ori目录下
    avg_cut_four_square_main(ori_path+'image/', im_list=image_list,suffix='png',save_path='E:/ML_data/remote_data/BJRoad/train_val_part/image/')
    avg_cut_four_square_main_gps(ori_path+'gps/', im_list=image_list,suffix='jpg',save_path='E:/ML_data/remote_data/BJRoad/train_val_part/gps/')
    avg_cut_four_square_main_mask(ori_path+'mask/', im_list=image_list,suffix='png',save_path='E:/ML_data/remote_data/BJRoad/train_val_part/mask/')
