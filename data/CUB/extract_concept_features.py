import torch
import open_clip
import numpy as np
import clip

"""
This snippet extracts the 312 text features (of the CUB_200 concepts) using the clip model and saves as a numpy file.
"""

device = torch.device("cuda:0")

# clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
# for p in clip_model.parameters(): p.requires_grad=False
# tokenizer = open_clip.get_tokenizer('ViT-B-16')

text_concepts = open("data/my_cub_concepts.txt").readlines()
for i in range(len(text_concepts)): text_concepts[i] = text_concepts[i].strip()
print("\n",len(text_concepts))

# tokenized_text_concepts = clip.tokenize(text_concepts)
tokenized_text_concepts = torch.cat([clip.tokenize(text_concept)
                           for text_concept in text_concepts]).to(device)
print("\n",tokenized_text_concepts.shape)

# device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load('ViT-L/14', device=device)


with torch.no_grad():
    # tokenized_text_concepts = tokenized_text_concepts.to(device)

    concept_features = clip_model.encode_text(tokenized_text_concepts)
    print("\n",concept_features.shape)

    np.save("my_cub_concept_features.npy",concept_features.cpu().numpy())