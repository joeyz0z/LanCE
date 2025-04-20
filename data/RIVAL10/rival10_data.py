import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
import glob
import json
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pickle
from binascii import a2b_base64
from tqdm import tqdm
import clip
from utils import *
from args import get_args

"""
Original Dataset() class for the RIVAL-10 dataset
Refer to https://github.com/mmoayeri/RIVAL10/blob/gh-pages/datasets/local_rival10.py 
"""

_DATA_ROOT = 'G:/DATA/DomainAdaptation/RIVAL10/{}/'
_LABEL_MAPPINGS = 'G:/DATA/DomainAdaptation/RIVAL10/meta/label_mappings.json'
_WNID_TO_CLASS = 'G:/DATA/DomainAdaptation/RIVAL10/meta/wnid_to_class.json'

_ALL_ATTRS = ['long-snout', 'wings', 'wheels', 'text', 'horns', 'floppy-ears',
              'ears', 'colored-eyes', 'tail', 'mane', 'beak', 'hairy',
              'metallic', 'rectangular', 'wet', 'long', 'tall', 'patterned']


def attr_to_idx(attr):
    return _ALL_ATTRS.index(attr)


def idx_to_attr(idx):
    return _ALL_ATTRS[idx]


def resize(img):
    return np.array(Image.fromarray(np.uint8(img)).resize((224, 224))) / 255


def to_3d(img):
    return np.stack([img, img, img], axis=-1)


def save_uri_as_img(uri, fpath='tmp.png'):
    ''' saves raw mask and returns it as an image'''
    binary_data = a2b_base64(uri)
    with open(fpath, 'wb') as f:
        f.write(binary_data)
    img = mpimg.imread(fpath)
    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img



class LocalRIVAL10(Dataset):
    def __init__(self, train=True, masks_dict=True):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.

        See __getitem__ for more documentation.
        '''
        self.train = train
        self.data_root = _DATA_ROOT.format('train' if self.train else 'test')
        self.masks_dict = masks_dict

        self.instance_types = ['ordinary']
        self.instances = self.collect_instances()
        self.resize = transforms.Resize((224, 224))

        with open(_LABEL_MAPPINGS, 'r') as f:
            self.label_mappings = json.load(f)
        with open(_WNID_TO_CLASS, 'r') as f:
            self.wnid_to_class = json.load(f)

    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self, ):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = self.data_root + subdir
            for f in tqdm(glob.glob(dir_path + '/*')):
                f = f.replace("\\", '/')
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    def transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8, 1.0), ratio=(0.75, 1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(self.resize(img))

            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)

            transformed_imgs.append(img)

        return transformed_imgs

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224, 224, 3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = save_uri_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path, merged_mask_path, mask_dict_path = self.all_instances[i]

        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load img
        img = Image.open(img_url)
        if img.mode == 'L':
            img = img.convert("RGB")

        # load labels
        labels = np.load(label_path)
        attr_labels = torch.Tensor(labels[0]).long()
        changed_attrs = torch.Tensor(labels[1]).long()  # attrs that were added or removed

        merged_mask_img = Image.open(merged_mask_path)
        imgs = [img, merged_mask_img]
        if self.masks_dict:
            try:
                with open(mask_dict_path, 'rb') as fp:
                    mask_dict = pickle.load(fp)
            except:
                mask_dict = dict()
            for attr in mask_dict:
                mask_uri = mask_dict[attr]
                mask = save_uri_as_img(mask_uri)
                imgs.append(Image.fromarray(np.uint8(255 * mask)))

        transformed_imgs = self.transform(imgs)
        img = transformed_imgs.pop(0)
        merged_mask = transformed_imgs.pop(0)
        out = dict({'img': img,
                    'attr_labels': attr_labels,
                    'changed_attrs': changed_attrs,
                    'merged_mask': merged_mask,
                    'og_class_name': class_name,
                    'og_class_label': class_label})
        if self.masks_dict:
            attr_masks = [torch.zeros(img.shape) for i in range(len(_ALL_ATTRS) + 1)]
            for i, attr in enumerate(mask_dict):
                # if attr == 'entire-object':
                ind = -1 if attr == 'entire-object' else attr_to_idx(attr)
                attr_masks[ind] = transformed_imgs[i]
            out['attr_masks'] = torch.stack(attr_masks)

        return out

class CBM_RIVAL10(Dataset):
    def __init__(self, args, data_root=None, split="train",meta_root=None, src_dm_texts=None,tgt_dm_texts =None):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.

        See __getitem__ for more documentation.
        '''
        self.data_root = os.path.join(data_root,split).replace('\\', '/')
        label_mappings_path = os.path.join(data_root, "meta/label_mappings.json").replace('\\', '/')
        wnid_to_class_path = os.path.join(data_root, "meta/wnid_to_class.json").replace('\\', '/')


        self.instance_types = ['ordinary']
        self.instances = self.collect_instances()
        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)
        self.classname2id = {
              "truck":0,"car":1,"plane":2,"ship":3,"cat":4,"dog":5,"equine":6,"deer":7,"frog":8,"bird":9
          }
        with open(os.path.join(meta_root,"rival10_concepts.txt"),"r") as f:
            self.concept2id = {x.rstrip():c_id for c_id, x in enumerate(f.readlines())}
        with open(os.path.join(meta_root,"class_avg_attribute.pkl"), "rb") as f:
            self.class_avg_attribute = pickle.load(f)
            self.cls_avg_concept = args.class_avg_concept

        with open(label_mappings_path, 'r') as f:
            self.label_mappings = json.load(f)
        with open(wnid_to_class_path, 'r') as f:
            self.wnid_to_class = json.load(f)
        if src_dm_texts is not None and tgt_dm_texts is not None:
            self.domain_diffs = []
            print("----------Computing Domain Differences----------")
            for src_prompts, tgt_prompts in tqdm(zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)):
                tqdm.write(tgt_prompts+" - "+src_prompts)
                source_embeddings, target_embeddings = get_domain_text_embs(self.clip_model, [src_prompts], [tgt_prompts],
                                                                            list(self.classname2id.keys()))
                source_embeddings /= source_embeddings.norm(dim=-1, keepdim=True)
                target_embeddings /= target_embeddings.norm(dim=-1, keepdim=True)
                diffs = target_embeddings.float() - source_embeddings.float()
                if diffs.norm() == 0:
                    print(diffs)
                diffs /= diffs.norm(dim=-1, keepdim=True)
                self.domain_diffs.append(diffs)
            self.domain_diffs = torch.cat(self.domain_diffs, dim=0).to(device)
        self.clip_model = None


    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self, ):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = os.path.join(self.data_root,subdir).replace('\\','/')
            for f in tqdm(glob.glob(dir_path + '/*')):
                f = f.replace("\\", '/')
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    instances.append((img_url, label_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224, 224, 3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = save_uri_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path = self.all_instances[i]
        image = Image.open(img_url).convert("RGB")
        image = self.preprocess(image)
        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load concept labels
        if self.cls_avg_concept == True:
            attr_labels = self.class_avg_attribute[class_label]
        else:
            all_attr_labels = np.load(label_path)
            attr_labels = torch.Tensor(all_attr_labels[0]).long()

        return image, class_label, attr_labels

class CBM_RIVAL10S(Dataset):
    def __init__(self, args, data_root, split, classname2id=None, concept2id=None,):

        if split == "train":
            raise Exception("No processed data for train split")
        elif split == "test":
            classnames = os.listdir(data_root)
        self.classname2id = {
              "truck":0,"car":1,"plane":2,"ship":3,"cat":4,"dog":5,"equine":6,"deer":7,"frog":8,"bird":9
          }
        self.all_instances = []
        for classname in classnames:
            class_id = self.classname2id[classname.lower()]
            for img_url in glob.glob(os.path.join(data_root, classname, "*.JPEG")):
                self.all_instances.append((img_url, class_id))


        device = args.device
        _, self.preprocess = clip.load(args.CLIP_type, device=device)

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        img_path, cls_label = self.all_instances[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.preprocess(image)

        # note that cls label starts from 1 and not 0
        label = int(cls_label)
        attr_label = torch.tensor([0]*18)

        return image, label, attr_label

"""
Explore dataset class
"""
if __name__ == '__main__':
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    from domain_prompt import source_text_prompts, target_text_prompts
    train_dataset = CBM_RIVAL10(args, data_root="G:/DATA/DomainAdaptation/RIVAL10", split="train",
                                meta_root="data/RIVAL10",
                                src_dm_texts=source_text_prompts, tgt_dm_texts=target_text_prompts)
    class2attribute = {}
    for idx in tqdm(range(len(train_dataset))):
        image, label, attr_label = train_dataset[idx]
        if label not in class2attribute:
            class2attribute[label] = [attr_label]
        else:
            class2attribute[label].append(attr_label)
    class_avg_attribute = {k: torch.stack(v, dim=0).float().mean(0) for k, v in class2attribute.items()}
    with open("data/RIVAL10/class_avg_attribute.pkl", "wb") as f:
        pickle.dump(class_avg_attribute, f)
    print(0)