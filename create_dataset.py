import os
import numpy as np
import shutil

'''
功能说明：
1. 开启数据集文件物理划分方式(FUNC=SPLIT_DATASET)
将原图目录和标签目录按照指定的划分比例（复制/移动）到指定的输出目录，并且在指定的输出目录自动创建如下的文件结构：
例如，指定输出目录为(output_dir)
--output_dir
    --images
        --train
        --val
        --test
    --labels
        --train
        --val
        --test
注：若是需要的文件名和程序默认的不一致，可以到split_data()函数内修改默认的文件名

2. 开启输出train.txt, val.txt, test.txt三个txt文件(FUNC=WRITE_TXT)
不移动源图像和标签的目录内的文件，按照指定的比例将划分后的训练集，测试集和验证集的文件路径分别输出到train.txt, val.txt, test.txt。
例如，指定输出目录为(output_dir)
--output_dir
    --train.txt
    --val.txt
    --test.txt

3. 注意事项
3.1 标签名和图像名务必一一对应，要求所有图像统一格式，所有标签统一格式。
3.2 为提高效率，本程序不支持遍历源目录中子目录下的文件，请将所有源文件都放到一级目录,否则会出问题
'''
'''
####################    输入参数设置(开始)    #################### 
'''
# 路径
# 源图像存储的目录：
org_images_dir = r'D:\dataset\fire1022\images'
# 源标签存储的目录（图像和标签的名称必须一一对应）：
org_labels_dir = r'D:\dataset\fire1022\labels'
# 输出的目标路径：
output_dir = r'D:\dataset\fire1022'

# 功能:
FUNC = 'SPLIT_DATASET'  # 划分后移动文件
# mode = 'copy' # 当FUNC = 'SPLIT_DATASET'时才生效，选择从源目录复制到新目录
mode = 'move'  # 当FUNC = 'SPLIT_DATASET'时才生效，选择从源目录移动到新目录

# FUNC = 'WRITE_TXT'          # 直接输出包含路径的3个txt文件

# 划分比例,依次是训练集比例，（三者之和必须等于一）
scale = [0.8, 0.15, 0.05]

# 文件格式设置（默认按照yolo格式）
# 图像类型
image_type = 'jpg'
# 标签类型
label_type = 'txt'
# output --data yaml
# (yolo专属) 直接输出yolo的data参数的yaml文件(要求输出路径必须是绝对路径，否则后面训练很容易找不到路径)，输出到output_dir
is_output_yaml = True
# 若输出yaml文件，必须提供类的信息
classes = ["fire", "coal", "hat", "person", "Roller"]  # class names

'''
####################    输入参数设置(结束)    #################### 
'''


def shuffle_file(org_images_dir):
    filenames = []
    # 遍历源图像目录
    for root, dir, files in os.walk(org_images_dir):
        for file in files:
            if file[-len(image_type):] == image_type:  # 判断图像格式
                label_name = file[:-len(image_type)] + label_type
                # 判断是否存在对应标签
                if os.path.exists(os.path.join(org_labels_dir, label_name)):
                    filenames.append(file)  # 保存文件名称

    # 打乱文件名列表
    np.random.shuffle(filenames)

    # 划分训练集、验证集，测试集
    if len(scale) != 3:
        print('划分比例设置有误，划分数组的元素量不为3，请检查')
        return False
    elif float(scale[0]) < 0 or float(scale[1]) < 0 or float(scale[1]) < 0:
        print('划分比例设置有误，存在划分参数<0，请检查')
        return False
    elif float(scale[0]) + float(scale[1]) + float(scale[2]) != 1:
        print('划分比例设置有误，划分比例总和不为1，请检查')
        return False

    return True, filenames


def split_data(train_val_test_set, mode, org_images_dir, org_labels_dir, output_dir):
    # 子目录文件名的设置默认按照yolo的要求，如果有差异可以修改
    img_label = ['images', 'labels']  # 图像目录名称和标签目录名称
    train_val_test = ['train', 'val', 'test']  # 子目录训练集，验证集，测试集的名称
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # 创建新目录
    for i in img_label:
        type_dir = os.path.join(output_dir, i)
        if not os.path.exists(type_dir):
            os.mkdir(type_dir)
        for j in train_val_test:
            split_dir = os.path.join(type_dir, j)
            if not os.path.exists(split_dir):
                os.mkdir(split_dir)

    # 移动/复制文件到新目录
    for i in range(len(img_label)):
        for j in range(len(train_val_test_set)):
            for k in range(len(train_val_test_set[j])):
                if i == 0:  # image
                    file_name = train_val_test_set[j][k]
                    old_path = os.path.join(org_images_dir, file_name)
                else:  # label
                    file_name = train_val_test_set[j][k][:-len(image_type)] + label_type
                    # print("train_val_test_set[j][k]=", train_val_test_set[j][k])
                    # print("file_name=", file_name)

                    old_path = os.path.join(org_labels_dir, file_name)
                new_path = os.path.join(
                    os.path.join(os.path.join(output_dir, img_label[i]), train_val_test[j]), file_name)
                if mode == 'copy':
                    shutil.copyfile(old_path, new_path)
                elif mode == 'move':
                    # print("old_path=",old_path)
                    # print("new_path=",new_path)
                    shutil.move(old_path, new_path)
                else:
                    print('mode设置错误, 划分数据集取消')
                    return
    if is_output_yaml:
        train_line = 'train: ' + os.path.join(os.path.join(output_dir, img_label[i]), train_val_test[0]) + '\n'
        val_line = 'val: ' + os.path.join(os.path.join(output_dir, img_label[i]), train_val_test[0]) + '\n\n'
        nc_line = 'nc: ' + str(len(classes)) + '\n'
        classes_line = 'names: ['
        for cls in range(len(classes)):
            class_msg = '\'' + classes[cls] + '\''
            classes_line += class_msg
            if cls < len(classes) - 1:
                classes_line += ','
        classes_line += ']'
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(train_line)
            f.write(val_line)
            f.write(nc_line)
            f.write(classes_line)


def write_txt(train_val_test_set, org_images_dir, output_dir):
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f1, \
            open(os.path.join(output_dir, 'val.txt'), 'w') as f2, \
            open(os.path.join(output_dir, 'test.txt'), 'w') as f3:

        path_set = []
        print("len(train_val_test_set)=", len(train_val_test_set))
        for i in range(len(train_val_test_set)):
            new_lines = []
            for j in range(len(train_val_test_set[i])):
                path = os.path.join(org_images_dir, train_val_test_set[i][j]) + '\n'
                new_lines.append(path)
            if i == 0:
                f1.writelines(new_lines)
            elif i == 1:
                f2.writelines(new_lines)
            elif i == 2:
                f3.writelines(new_lines)
    if is_output_yaml:
        train_line = 'train: ' + os.path.join(output_dir, 'train.txt') + '\n'
        val_line = 'val: ' + os.path.join(output_dir, 'val.txt') + '\n\n'
        nc_line = 'nc: ' + str(len(classes)) + '\n'
        classes_line = 'names: ['
        for cls in range(len(classes)):
            class_msg = '\'' + classes[cls] + '\''
            classes_line += class_msg
            if cls < len(classes) - 1:
                classes_line += ','
        classes_line += ']'
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(train_line)
            f.write(val_line)
            f.write(nc_line)
            f.write(classes_line)


if __name__ == "__main__":

    ret, filenames = shuffle_file(org_images_dir)
    if ret:
        train = filenames[:int(len(filenames) * scale[0])]
        val = filenames[int(len(filenames) * scale[0]):int(len(filenames) * scale[0] + len(filenames) * scale[1])]
        test = filenames[int(len(filenames) * scale[0] + len(filenames) * scale[1]):]
        train_val_test_set = [train, val, test]
        if FUNC == 'SPLIT_DATASET':
            split_data(train_val_test_set, mode, org_images_dir, org_labels_dir, output_dir)
        elif FUNC == 'WRITE_TXT':
            write_txt(train_val_test_set, org_images_dir, output_dir)
