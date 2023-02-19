import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--file_name', help='image list in file', required=True)
    parser.add_argument('--dest_path', type=str, default='')
    parser.add_argument("--filter", action="store_true", help="filter jinglianwen image list")
    args = parser.parse_args()

    return args

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)


def read_write_file_name(file_name, image_path, dest_path="./image_concat/"):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    image_id = 0
    for image_name in file_lists:
        image_before, image_current, image_after, lr = image_name.split()

        image_write = os.path.join(dest_path, image_before.split('/')[-1] + "_" + image_current.split('/')[-1] + "_" + image_after.split('/')[-1] + str(image_id) + ".jpg")
        print("write current image: ", image_write)
        MkdirSimple(image_write)
        image_id += 1

        image_before = os.path.join(image_path, image_before)
        image_current = os.path.join(image_path, image_current)
        image_after = os.path.join(image_path, image_after)
        img_before = cv2.imread(image_before)
        img_current = cv2.imread(image_current)
        img_after = cv2.imread(image_after)
        image = np.hstack([img_before, img_current, img_after])
        MkdirSimple(image_write)
        cv2.imwrite(image_write, image)

def filter_jinglianwen_uncontinue(file_name, dest_path=""):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    need_remove_diff_second = 0
    need_remove_diff_minute = 0

    if len(file_lists) > 1:
        MkdirSimple(os.path.join(dest_path, "same_minute_same_second.txt"))

        for idx in tqdm(range(len(file_lists))):
            images = file_lists[idx].split()

            image_0 = images[0]
            image_1 = images[1]
            image_2 = images[2]
            before = os.path.join(*(image_0.split('_')[:-2]))
            current = os.path.join(*(image_1.split('_')[:-2]))
            after = os.path.join(*(image_2.split('_')[:-2]))

            if before != current or current != after:
                print("diff_minute: ", before, current, after)
                with open(os.path.join(dest_path, "diff_minute.txt"), 'a') as f:
                    f.write(file_lists[idx])
                need_remove_diff_minute += 1
                continue
            else:
                with open(os.path.join(dest_path, "same_minute.txt"), 'a') as f:
                    f.write(file_lists[idx])
                second_before = image_0.split('_')[-2]
                second_current = image_1.split('_')[-2]
                second_after = image_2.split('_')[-2]
                if abs(int(second_current) - int(second_before))%60 > 1 or abs(int(second_after) - int(second_current))%60 > 1:
                    need_remove_diff_second += 1
                    with open(os.path.join(dest_path, "same_minute_diff_second.txt"), 'a') as f:
                        f.write(file_lists[idx])
                    print("same_minute_diff_second")
                    continue
                else:
                    print("same_minute_same_second")
                    with open(os.path.join(dest_path, "same_minute_same_second.txt"), 'a') as f:
                        f.write(file_lists[idx])
                    pass

    print("same_minute_diff_second data: ", need_remove_diff_second)
    print("diff_minute", need_remove_diff_minute)


if __name__ == '__main__':

    args = GetArgs()
    if (args.filter):
        filter_jinglianwen_uncontinue(args.file_name, args.dest_path)
    else:
        img_list = read_write_file_name(args.file_name, args.data_path, args.dest_path)
