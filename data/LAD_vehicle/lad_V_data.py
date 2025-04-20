from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import clip
import pickle
import os
from tqdm import tqdm
from utils import *
from args import get_args

class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            self.data.extend(pickle.load(open(file_path, 'rb')))
        _, self.preprocess = clip.load('ViT-L/14', device=device)

        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
                img_path = img_path.replace('images/', '')
            else:
                img_path = "F:/Optimus/domain_concep/new-project/myCBM-v1/data/awa2"+'/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        return img, class_label, attr_label



class Processed_LADV_Labo(Dataset):
    def __init__(self, args, data_root, split, meta_root=None,attr_name=None,src_dm_texts=None,tgt_dm_texts =None):

        if split == "train":
            self.img_features = torch.load(os.path.join(data_root, "img_feat_train_all_00_ViT-L-14.pth"))
            self.img_labels = torch.load(os.path.join(data_root, "label_train_all.pth"))
        elif split == "test":
            self.img_features = torch.load(os.path.join(data_root, "img_feat_test_00_ViT-L-14.pth"))
            self.img_labels = torch.load(os.path.join(data_root, "label_test.pth"))

        self.data_root = data_root
        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)

        with open(r"data/LAD_vehicle/class4zs.txt", "r") as f:
            self.classname2id = {x.split(" ", 1)[1].rstrip(): (int(x.split(" ", 1)[0]) - 1) for x in f.readlines()}
        with open(r"data/LAD_vehicle/LAD_vehicle_concept.txt","r",encoding="utf-8") as f:
            self.concept2id = {x.rstrip():c_id for c_id, x in enumerate(f.readlines())}

        # concept_labels
        if src_dm_texts is not None and tgt_dm_texts is not None:
            self.domain_diffs = []
            print("----------Computing Domain Differences----------")
            for src_prompts, tgt_prompts in tqdm(zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)):
                tqdm.write(tgt_prompts+" - "+src_prompts)
                source_embeddings, target_embeddings = get_domain_text_embs(self.clip_model, [src_prompts], [tgt_prompts],
                                                                            list(self.classname2id.keys()),device)
                source_embeddings /= source_embeddings.norm(dim=-1, keepdim=True)
                target_embeddings /= target_embeddings.norm(dim=-1, keepdim=True)
                diffs = target_embeddings.float() - source_embeddings.float()
                if diffs.norm() == 0:
                    print(diffs)
                diffs /= diffs.norm(dim=-1, keepdim=True)
                self.domain_diffs.append(diffs)
            self.domain_diffs = torch.stack(self.domain_diffs, dim=0).to(device)
        self.clip_model = None

    def __len__(self):
        return len(self.img_features)

    def __getitem__(self, idx):
        image = self.img_features[idx]
        label = self.img_labels[idx]
        attr_label = torch.tensor([0] * len(self.concept2id))
        return image, label, attr_label

class Processed_LADV(Dataset):
    def __init__(self, args, data_root, split, meta_root=None,attr_name=None,src_dm_texts=None,tgt_dm_texts =None):

        if split == "train":
            self.annos = open(os.path.join(meta_root, "train_vehicle.txt")).readlines()
        elif split == "test":
            self.annos = open(os.path.join(meta_root, "test_vehicle.txt")).readlines()
        else:
            raise ValueError("split must be train or test")
        with open(r"data/LAD_vehicle/class_attribute_avg.pkl", "rb") as f:
            self.class_avg_attribute = pickle.load(f)
            self.cls_avg_concept = args.class_avg_concept
        self.data_root = data_root
        device = args.device
        self.clip_model, self.preprocess = clip.load(args.CLIP_type, device=device)
        # with open(r"data/LAD_vehicle/class.txt", "r") as f:
        #     self.classname2id = {x.split(" ")[1].rstrip(): (int(x.split(" ")[0]) - 1)for x in f.readlines()}
        with open(r"data/LAD_vehicle/class4zs.txt", "r") as f:
            self.classname2id = {x.split(" ", 1)[1].rstrip(): (int(x.split(" ", 1)[0]) - 1) for x in f.readlines()}

            # self.classname2id = {x.split(" ")[1][4:].replace("_"," ").lower().rstrip():(int(x.split(" ")[0])-1) for x in f.readlines()}
        with open(r"data/LAD_vehicle/LAD_vehicle_conceptNet_concepts.txt","r") as f:
            self.concept2id = {x.rstrip():c_id for c_id, x in enumerate(f.readlines())}

        # concept_labels
        # attribute_path = os.path.join(meta_root, attr_name)
        attribute_path = None
        if attribute_path is not None:
            with open(attribute_path,'rb') as f:
                self.path2attribute = pickle.load(f)
        else:
            self.path2attribute = None
        if src_dm_texts is not None and tgt_dm_texts is not None:
            self.domain_diffs = []
            print("----------Computing Domain Differences----------")
            for src_prompts, tgt_prompts in tqdm(zip(src_dm_texts * len(tgt_dm_texts), tgt_dm_texts)):
                tqdm.write(tgt_prompts+" - "+src_prompts)
                source_embeddings, target_embeddings = get_domain_text_embs(self.clip_model, [src_prompts], [tgt_prompts],
                                                                            list(self.classname2id.keys()),device)
                source_embeddings /= source_embeddings.norm(dim=-1, keepdim=True)
                target_embeddings /= target_embeddings.norm(dim=-1, keepdim=True)
                diffs = target_embeddings.float() - source_embeddings.float()
                if diffs.norm() == 0:
                    print(diffs)
                diffs /= diffs.norm(dim=-1, keepdim=True)
                self.domain_diffs.append(diffs)
            self.domain_diffs = torch.stack(self.domain_diffs, dim=0).to(device)
        self.clip_model = None

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label, top_x, top_y, btm_x, btm_y = self.annos[idx].strip().split(",")[0:6]
        image = Image.open(os.path.join(self.data_root,img_path)).convert("RGB")
        image = self.preprocess(image)
        # note that cls label starts from 1 and not 0
        label = int(cls_label)
        # self.cls_avg_concept == True
        # if self.cls_avg_concept == True:
        #     attr_label = self.class_avg_attribute[label]
        # else:
        #     attr_label = torch.tensor(self.path2attribute[img_path])
        attr_label = torch.tensor([0] * 221)
        return image, label, attr_label

class Processed_LADV3(Dataset):
    def __init__(self, args, data_root, split, meta_root=None, classname2id=None,concept2id=None):
        ###


        if split == "train":
            raise Exception("No processed data for train split")
        elif split == "test":
            self.annos = open(os.path.join(meta_root, "test_CAD_vehicles.txt")).readlines()
        self.data_root = data_root
        device = args.device
        _, self.preprocess = clip.load(args.CLIP_type, device=device)
        self.classname2id = concept2id
        self.concept2id = concept2id


    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        img_path, cls_label = self.annos[idx].strip().split(",")[0:2]

        image = Image.open(os.path.join(self.data_root,img_path)).convert("RGB")
        image = self.preprocess(image)
        label = int(cls_label)
        ## target source has no concept label
        attr_label = torch.tensor([0]*221)

        return image, label, attr_label

class Processed_LADV3_Labo(Dataset):
    def __init__(self, args, data_root, split, meta_root=None, classname2id=None,concept2id=None):
        ###
        if split == "train":
            raise Exception("No processed data for train split")
        elif split == "test":
            self.img_features = torch.load(os.path.join(data_root, "img_feat_test_00_ViT-L-14.pth"))
            self.img_labels = torch.load(os.path.join(data_root, "label_test.pth"))
        self.data_root = data_root
        device = args.device
        _, self.preprocess = clip.load(args.CLIP_type, device=device)
        self.classname2id = concept2id
        self.concept2id = concept2id


    def __len__(self):
        return len(self.img_features)

    def __getitem__(self, idx):
        image = self.img_features[idx]
        label = self.img_labels[idx]
        attr_label = torch.tensor([0] * len(self.concept2id))
        return image, label, attr_label

if __name__ == '__main__':
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    from domain_prompt import source_text_prompts, target_text_prompts

    train_dataset = Processed_LADV(args,data_root="/mnt/20ee3a83-3f42-40a7-b9b4-cfc9beade036/zzq/DomainData/LAD_data/vehicles",
                                          split="train",
                                          meta_root="data/LAD_vehicle",
                                          attr_name=None,
                                          src_dm_texts=source_text_prompts, tgt_dm_texts=target_text_prompts)
    class2attribute = {}
    for idx in tqdm(range(len(train_dataset))):
        image, label, attr_label = train_dataset[idx]
        if label not in class2attribute:
            class2attribute[label] = [attr_label]
        else:
            class2attribute[label].append(attr_label)
    class_avg_attribute = {k: torch.stack(v, dim=0).float().mean(0) for k, v in class2attribute.items()}
    with open("data/LAD_vehicle/class_avg_attribute.pkl", "wb") as f:
        pickle.dump(class_avg_attribute, f)
    print(0)

