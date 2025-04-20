from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import clip
import pickle
import os
from tqdm import tqdm


"""
Returns image and class-label
Used to train vanilla CLIP VL-CBM
"""
def zeroshot_classifier(prompts, model, normalize=True, model_type='clip', device="device"):
    """ Computes CLIP text embeddings for a list of prompts. """
    model.eval()
    assert type(prompts) == list, "prompts must be a list"
    with torch.no_grad():
        zeroshot_weights = []
        texts = clip.tokenize(prompts).to(device) # tokenize
        # texts = clip.tokenize(prompts).cuda() #tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        if normalize:
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        # class_embedding = class_embeddings.mean(dim=0)
    return class_embeddings.cpu()

def get_domain_text_embs(model, source_text_prompts, target_text_prompts, class_names, device):
    """
    Gets the text embeddings of the prompts describing the source and target domains.
    If generic is True, source_text_prompts and target_text_prompts are strings instead of
    templates to put the class name in.
    """
    if False:
        ## class-agnostic
        text_embeddings = zeroshot_classifier(target_text_prompts, model, normalize=True, model_type="clip")
        text_embeddings = np.transpose(text_embeddings, (1, 0))
        orig_prompts = text_embeddings
        if len(source_text_prompts) > 0:
            source_embeddings = zeroshot_classifier(source_text_prompts, model, normalize=True, model_type="clip")
            print("source emb before averaging", source_embeddings.shape)
            source_embeddings = source_embeddings.mean(dim=0)
            print("source emb after averaging", source_embeddings.shape)
            diffs = torch.stack([emb - source_embeddings[0] for emb in text_embeddings])
            diffs /= text_embeddings.norm(dim=-1, keepdim=True)
    else:
        templates = target_text_prompts
        all_texts = []
        for t in source_text_prompts:
            texts = [t.format(c) for c in class_names]
            text_emb = zeroshot_classifier(texts, model, normalize=False, model_type="clip",device = device)
            all_texts.append(text_emb)
        for p in target_text_prompts:
            texts = [p.format(c) for c in class_names]
            text_emb = zeroshot_classifier(texts, model, normalize=False, model_type="clip",device = device)
            all_texts.append(text_emb)
        # this subtracts the neutral embedding from the domain embeddings and normalizes.
    return all_texts[0], all_texts[1]