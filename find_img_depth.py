import os
import argparse

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="image dir")
    parser.add_argument("--output", type=str, default=None, help="output file list")

    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = GetArgs()

    dl_dir = args.input # "/data/ABBY/BASE/TEST/depth/data_2023_08_03/data_2023_08_03_2"
    file_save = args.output
    dl_image_list = os.path.join(dl_dir, "image_list.txt")
    depth_dir = os.path.join(dl_dir, "depth_data3")
    if not os.path.exists(depth_dir):
        depth_dir = os.path.join(dl_dir, "depth_data2")
        if not os.path.exists(depth_dir):
            print("depth image path :{} not exist end!".format(depth_dir))

    with open(dl_image_list, 'r') as f:
        image_list = f.readlines()
        print(image_list)
    depth_all_file = []
    depth_not_right = []
    depth_left_file_write = []
    for root, dirs, files in os.walk(depth_dir):
        for file in files:
            depth_all_file.append(os.path.join(root, file))
    print(depth_all_file)
    depth_number = 0
    dl_number = 0

    for temp in depth_all_file:
        if "right" in temp and "png" in temp:
            continue
        else:
            depth_not_right.append(temp)
    depth_not_right.sort()
    image_list.sort()

    while depth_number < len(depth_not_right) and dl_number < len(image_list):
        if ("left" in depth_not_right[depth_number]) and "png" in depth_not_right[depth_number]:
            print("dl_number: ", dl_number)
            print(image_list[dl_number])
            dl_time = image_list[dl_number].split('/')[-1].split('.')[0].split('_')[1]
            print(dl_time)
            depth_time = depth_not_right[depth_number].split('/')[-1].split(".")[0][3:]
            print(depth_time)
            distance = int(depth_time) - int(dl_time)
            print(distance)

            distance = distance /1000000.0
            if (depth_time == "1691054138463070"):
                print("DDDDistance: ", distance)
                print(len(depth_left_file_write))
            print(distance)
            if (distance > 0.2):
                dl_number += 1
                continue
            elif (distance < -0.2):
                depth_number += 1
                continue
            else:
                depth_left_file_write.append(depth_not_right[depth_number])
                depth_number += 1
        else:
            depth_left_file_write.append(depth_not_right[depth_number])
            depth_number += 1
    print(len(depth_left_file_write), len(image_list))
    depth_all_write = []
    for file in depth_left_file_write:
        depth_all_write.append(file)
        if "left" in file:
            depth_all_write.append(file.replace("left", "right"))
    print(depth_all_write)
    with open(args.output, "w") as f:
        for file in depth_all_write:
            f.write(file)
            f.write("\n")

    print(depth_left_file_write[19])
    print(depth_left_file_write[20])