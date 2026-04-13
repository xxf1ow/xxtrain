import argparse
import os

import PIL.Image
from tqdm import tqdm

from .annparser import find_dir, find_img, parse_labelimg

# 保存数据集中出现的不在允许列表中的标签, 用于最后检查允许列表是否正确
skip_categories = set()
skip_files = set()


# 单个图片
def generate(img_path, xml_path, txt_path, categories):
    # check image
    assert os.path.isfile(img_path), f'图片文件不存在: {img_path}'
    img = PIL.Image.open(img_path)
    width, height = img.size
    assert width > 0 and height > 0
    # parse labelimg anns file
    bbox_dict = parse_labelimg(xml_path, width, height)
    # generate anns
    boxes = []
    for instance, box in bbox_dict.items():
        label = instance[0]
        if label not in categories:
            skip_files.add(img_path)
            skip_categories.add(label)
            continue
        label_id = categories.index(label)
        x_center = (box[0] + box[2]) / 2.0 / width
        y_center = (box[1] + box[3]) / 2.0 / height
        w_norm = (box[2] - box[0]) / width
        h_norm = (box[3] - box[1]) / height
        boxes.append(f'{label_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(boxes))
    return len(boxes)


def process(root_path, split, reserve_no_label=True):
    print('\n[info] start task...')
    # 统计
    images_count = [0, 0]
    labels_count = [0, 0]
    # 目录检查
    work_path = os.path.join(root_path, 'images')
    save_path = os.path.join(root_path, 'labels')
    assert os.path.isdir(work_path), f'数据集不存在: {work_path}'
    os.makedirs(save_path, exist_ok=True)
    # 读取标签列表
    label_path = os.path.join(root_path, 'labels.txt')
    assert os.path.isfile(label_path), f'标签列表不存在: {work_path}'
    with open(label_path, encoding='utf-8') as f:
        categories = [line.strip() for line in f.readlines()]
    assert len(categories) > 0, f'标签列表为空: {label_path}'
    # 遍历子文件夹
    train_list = []
    test_list = []
    for dir in find_dir(work_path):
        imgs_dir_path = os.path.join(work_path, dir, 'imgs')
        if not os.path.isdir(imgs_dir_path):
            continue
        img_list = find_img(imgs_dir_path)
        not_ann_cnt = 0
        os.makedirs(os.path.join(save_path, dir, 'imgs'), exist_ok=True)
        for num, file in enumerate(tqdm(img_list, desc=f'{dir}\t', leave=True, ncols=100, colour='CYAN')):
            # misc path
            raw_name, extension = os.path.splitext(file)
            img_path = f'{work_path}/{dir}/imgs/{raw_name}{extension}'
            xml_path = f'{work_path}/{dir}/anns/{raw_name}.xml'
            txt_path = f'{save_path}/{dir}/imgs/{raw_name}.txt'
            # 生成标注文件
            ann_count = generate(img_path, xml_path, txt_path, categories)
            # 统计
            not_ann_cnt += 1 if ann_count == 0 else 0
            if reserve_no_label is False and ann_count == 0:
                continue
            if split <= 0 or num % split != 0:  # 训练集
                images_count[0] += 1
                labels_count[0] += ann_count
                train_list.append(img_path)
            if split <= 0 or num % split == 0:  # 测试集
                images_count[1] += 1
                labels_count[1] += ann_count
                test_list.append(img_path)
        if not_ann_cnt != 0:
            print(f'\033[1;31m[Error] {dir}中有{not_ann_cnt}张图片不存在标注文件\n\033[0m')
    print(f'\n训练集图片总数: {images_count[0]}, 标注总数: {labels_count[0]}\n')
    print(f'测试集图片总数: {images_count[1]}, 标注总数: {labels_count[1]}\n')
    with open(os.path.join(root_path, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_list))
    with open(os.path.join(root_path, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_list))
    content = f'path: {root_path}\ntrain: train.txt\nval: val.txt\n\nnames:\n'
    for i, name in enumerate(categories):
        content += f'  {i}: {name}\n'
    with open(os.path.join(root_path, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='VOC to YOLO format converter')
    args.add_argument('--root_path', type=str, required=True, help='数据集根目录')
    args.add_argument('--split', type=int, default=5, help='测试集划分比例, 例如5表示每5张图片划分为1张测试集')
    args.add_argument('--reserve_no_label', action='store_true', help='是否保留没有标注的图片')
    args = args.parse_args()
    process(args.root_path, args.split, args.reserve_no_label)
    if len(skip_categories) > 0:
        print(f'\n\033[1;33m[Warning] 出现但不被允许的标签: \033[0m{skip_categories}\n')
        print('\n'.join(sorted(skip_files)))
