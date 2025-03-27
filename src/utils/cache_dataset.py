import csv
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--d_path', default= '../datasets/cvat_cracks/', type=str, help='Relative path to the dataset you want to cache into CSV')
argparser.add_argument('--dataset_type', default= 'orig', type=str, choices=['orig', 'camvid'], help='Dataset type. Options: [orig, camvid]. Default: "orig"')

def append_csv(root):
    '''
    Appends a csv that will contain [image, mask] pairs per row of the csv. The csv is saved in the root of the directory. Assuming that `images` contain images and `masks` contain corresponding masks (with the same name as the image). 
    Args:
        root: root of the dataset. This is where the csv is going to be saved.
    '''
    img_path  = os.path.join(root, 'images')
    mask_path = os.path.join(root, 'masks')
    imgs = sorted(os.listdir(img_path))
    masks = sorted(os.listdir(mask_path))

    x = [os.path.join('images', sample) for sample in imgs]
    y = [os.path.join('masks', sample) for sample in masks]

    res = list(zip(x,y))

    #quick check whether the zip is done correctly
    for i in res:
        a = i[0].split('/')[-1].split('.')[0]
        b = i[1].split('/')[-1].split('.')[0]
        if a != b:
            print(f'{a} should be the same as {b}')

    csv_dest = os.path.join(root, 'image_mask_pairs.csv')
    print(csv_dest)

    with open(csv_dest, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(res)


if __name__ == "__main__":
    args = argparser.parse_args()
    dataset_root = args.d_path
    if args.dataset_type == 'orig':
        append_csv(dataset_root)
