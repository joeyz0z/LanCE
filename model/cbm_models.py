
import torch
import torchvision
import torch.nn as nn
import clip
import numpy as np
from tqdm import tqdm
import torch.nn.init as init

class Standard_cbm(nn.Module):
    def __init__(self, init_with_class_embedding=False):
        super(Standard_cbm, self).__init__()
        # clip model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, preprocess = clip.load('ViT-L/14', device=device)
        for p in self.clip_model.parameters(): p.requires_grad = False

        self.concept_embeddings = nn.Parameter(torch.empty(len(rival_atributes), 768))
        # 使用 Xavier 均匀初始化
        init.xavier_uniform_(self.concept_embeddings)

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(len(rival_atributes)),
            nn.Linear(len(rival_atributes), 10)
        )

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        dev = visual_features.device
        concept_activations = visual_features @ self.concept_embeddings.T  # (bs,312)
        return concept_activations, self.classifier(concept_activations)

class clipzs(nn.Module):
    def __init__(self, args, prompts, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clipzs, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        class_texts = [[prompt.format(c) for prompt in prompts] for c in class_names]
        class_embeddings = []
        for texts in class_texts:
            texts_ = clip.tokenize(texts).to(self.device)
            class_embedding = self.clip_model.encode_text(texts_).float()
            class_embedding /= class_embedding.norm()
            class_embedding = class_embedding.mean(0)[None]
            class_embeddings.append(class_embedding)
        self.class_embeddings = torch.cat(class_embeddings)
        self.class_embeddings /= self.class_embeddings.norm()

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        sim = visual_features @ self.class_embeddings.T  # (bs,312)
        return sim


class clip_mlp(nn.Module):
    def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_mlp, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)
        # classifer
        self.classifier = nn.Linear(feature_dim, len(class_names))

    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)

        return None, self.classifier(visual_features), None

    def extract_cls_concept(self):

        return {}, {}

class clip_mlp_orth(nn.Module):
    def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_mlp_orth, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)
        # classifer
        self.mapper = nn.Linear(feature_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, len(class_names))


    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        auxiliary_visual_features = self.mapper(visual_features)
        regularizer = self.mapper(self.diffs)

        return None, self.classifier(auxiliary_visual_features), regularizer

    def extract_cls_concept(self):
        return {}, {}

class clip_cbm(nn.Module):
    def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_cbm, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(len(concept_names)),
            nn.Linear(len(concept_names), len(class_names)))

        if init_with_class_embedding == True:
            with torch.no_grad():
                class_embeddings = self.clip_model.encode_text(clip.tokenize(class_names).to(device)).float()
                self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                self.classifier[2].weight = nn.Parameter(class_embeddings @ self.concept_embeddings.T)


    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()  # (bs,512)
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        concept_activations = visual_features @ self.concept_embeddings.T  # (bs,312)
        return concept_activations, self.classifier(concept_activations), self.diffs

    def extract_cls_concept(self):
        asso_mat_last = self.classifier[2].weight
        topk_res_last = {}
        topk_res = {}
        # import pdb; pdb.set_trace()
        concept_names = np.array(self.concept_names)
        for i, cls_name in enumerate(self.class_names):

            topk_res_last[cls_name] = np.unique(concept_names[asso_mat_last[i].topk(10)[1].cpu().detach().numpy()]).tolist()

        return {},topk_res

class clip_cbm_orth(nn.Module):
    def __init__(self, args, class_names, concept_names, domain_diffs,init_with_class_embedding=False):
        super(clip_cbm_orth, self).__init__()
        self.diffs = domain_diffs.clone()
        self.device = args.device
        self.class_names = class_names
        self.concept_names = concept_names
        self.clip_model, preprocess = clip.load(args.CLIP_type, device=self.device)
        feature_dim = self.clip_model.visual.output_dim
        for p in self.clip_model.parameters(): p.requires_grad = False

        # concept_embeddings
        attr_texts = clip.tokenize(concept_names).to(self.device) # tokenize
        self.concept_embeddings = self.clip_model.encode_text(attr_texts).float()  # embed with text encoder
        self.concept_embeddings /= self.concept_embeddings.norm(dim=-1, keepdim=True)

        # classifer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(len(concept_names)),
            nn.Linear(len(concept_names), len(class_names)))

        if init_with_class_embedding == True:
            with torch.no_grad():
                class_embeddings = self.clip_model.encode_text(clip.tokenize(class_names).to(device)).float()
                self.class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                self.classifier[2].weight = nn.Parameter(class_embeddings @ self.concept_embeddings.T)


    def forward(self, images):
        visual_features = self.clip_model.encode_image(images).float()
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        concept_activations = visual_features @ self.concept_embeddings.T  # (bs,312)
        regularizer = self.classifier[1:](self.diffs @ self.concept_embeddings.T)
        return concept_activations, self.classifier(concept_activations), regularizer

    def extract_cls_concept(self):
        asso_mat_last = self.classifier[2].weight
        topk_res_last = {}
        topk_res = {}
        # import pdb; pdb.set_trace()
        concept_names = np.array(self.concept_names)
        for i, cls_name in enumerate(self.class_names):
            topk_res_last[cls_name] = np.unique(concept_names[asso_mat_last[i].topk(10)[1].cpu().detach().numpy()]).tolist()
        return {},topk_res

    


