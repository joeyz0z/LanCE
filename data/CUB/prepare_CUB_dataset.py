"""
Make train, val, test datasets based on train_test_split.txt, and by sampling val_ratio of the official train data to make a validation set 
Each dataset is a list of metadata, each includes official image id, full image path, class label, attribute labels, attribute certainty scores, and attribute labels calibrated for uncertainty
"""
import os
import random
import pickle
import argparse
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
from tqdm import tqdm


def extract_data(data_dir):
    cwd = os.getcwd()
    data_path = join(cwd,data_dir + '/images')

    path_to_id_map = dict() #map from full image path to image id
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[items[1]] = int(items[0])

    attribute_labels_all = ddict(list) #map from image id to a list of attribute labels
    with open(join(cwd, data_dir + '/attributes/image_attribute_labels.txt'), 'r') as f:
        for line in tqdm(f):
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_labels_all[int(file_idx)].append(attribute_label)

    is_train_test = dict() #map from image id to 0 / 1 (1 = train)
    with open(join(cwd, data_dir + '/train_test_split.txt'), 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print("Number of train images from official train test split:", sum(list(is_train_test.values())))

    train_data, test_data = [], []
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort() #sort by class index
    path_to_attribute_label = {}
    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]
        #classfile_list.sort()
        for cf in classfile_list:
            cls_path = folder_path.split('\\')[-1]
            img_id = path_to_id_map[join(cls_path, cf).replace("\\","/")]
            img_path = join(folder_path, cf)
            path_to_attribute_label[join(cls_path, cf).replace("\\","/")] = attribute_labels_all[img_id]
            metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                      'attribute_label': attribute_labels_all[img_id]}
            if is_train_test[img_id]:
                train_data.append(metadata)
            else:
                test_data.append(metadata)

    return train_data, test_data, path_to_attribute_label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets',default="CUBP_processed/CUB")
    parser.add_argument('-data_dir', help='Where to load the datasets',default="G:/DATA/DomainAdaptation/CUB/CUB_200_2011")
    args = parser.parse_args()
    train_data, test_data, path_to_attribute_label = extract_data(args.data_dir)
    f = open(args.save_dir + 'path2attr.pkl', 'wb')
    pickle.dump(path_to_attribute_label, f)
    f.close()
    for dataset in ['train','test']:
        print("Processing %s set" % dataset)
        f = open(args.save_dir + dataset + '.pkl', 'wb')
        if 'train' in dataset:
            pickle.dump(train_data, f)
        else:
            pickle.dump(test_data, f)
        f.close()

