import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import sys
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import transforms
import clip
from PIL import Image
from os import listdir, path, mkdir
import pandas as pd
import random
from scipy.stats import skew, spearmanr, pearsonr
from matplotlib import pyplot as plt

#Code for extracting SLIP representations is based on this Github response: https://github.com/facebookresearch/SLIP/issues/2
#Thanks to the authors.

def normalize(img, input_range = None):
    if input_range is None:
        minv = img.min()
    else:
        minv = input_range[0]
    img = img - minv

    if input_range is None:
        maxv = img.max()
    else:
        maxv = input_range[1] - minv

    if maxv != 0:
        img = img / maxv

    return img

def adjust_range(img, out_range, input_range = None):
    img = normalize(img, input_range = input_range)
    img = img * (out_range[1] - out_range[0])
    img = img + out_range[0]
    return img

class CLIP_Base():
    # Default CLIP model from OpenAI
    def __init__(self, model):
        self.device = "cuda"
        self.model  = model.eval()

        self.preprocess_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

    def preprocess(self, imgs, input_range = None):
        imgs = adjust_range(imgs, [0.,1.], input_range = input_range)
        return self.preprocess_transform(imgs)

    def encode_img(self, imgs, input_range = None, apply_preprocess = True):
        if apply_preprocess:
            imgs = self.preprocess(imgs, input_range = None)
        img_embeddings = self.model.encode_image(imgs)
        return img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)

    def encode_text(self, texts):
        text_embeddings = torch.stack([self.model.encode_text(clip.tokenize(text).to(self.device)).detach().clone() for text in texts])
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

sys.path.append('/home/autumn/SLIP')
import models
from tokenizer import SimpleTokenizer
import utils

class SLIP_Base():
    def __init__(self, model_name):
        self.device = "cuda"

        if model_name == "SLIP_VITB16":
            ckpt_path  = "/home/SLIP/slip_base_100ep.pt"

        self.preprocess_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.tokenizer = SimpleTokenizer()

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        # create model
        old_args = ckpt['args']
        old_args.model = model_name

        model = getattr(models, old_args.model)(rand_embed=False,
            ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        model.cuda().requires_grad_(False).eval()
        model.load_state_dict(state_dict, strict=True)

        n_params = sum(p.numel() for p in model.parameters())
        print("Loaded perceptor %s: %.2fM params" %(model_name, (n_params/1000000)))

        self.model = utils.get_model(model)

    def preprocess(self, imgs, input_range = None):
        #imgs = adjust_range(imgs, [0.,1.], input_range = input_range)
        return self.preprocess_transform(imgs)

    def encode_img(self, imgs, input_range = None, apply_preprocess = True):
        if apply_preprocess:
            imgs = self.preprocess(imgs, input_range = input_range)

        image_features = self.model.encode_image(imgs)
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        texts = self.tokenizer(texts).cuda(non_blocking=True)
        texts = texts.view(-1, 77).contiguous()
        text_embeddings = self.model.encode_text(texts)
        #text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings.unsqueeze(1)


def get_clip_perceptor(clip_model_name):
    if clip_model_name in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']:
        perceptor, preprocess = clip.load(clip_model_name, jit=False, device = "cuda")
        perceptor = perceptor.requires_grad_(False).eval()

        n_params = sum(p.numel() for p in perceptor.parameters())
        print("Loaded CLIP %s: %.2fM params" %(clip_model_name, (n_params/1000000)))
        clip_perceptor = CLIP_Base(perceptor, preprocess)

    else:
        clip_perceptor = SLIP_Base(clip_model_name)

    return clip_perceptor

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SLIP_Base('SLIP_VITB16')

preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


states_dict = {
    'Alaska':'ak',
    'Alabama':'al',
    'Arkansas':'ar',
    'Arizona':'az',
    'California':'ca',
    'Colorado':'co',
    'Connecticut':'ct',
    'Delaware':'de',
    'Florida':'fl',
    'Georgia':'ga',
    'Hawaii':'hi',
    'Iowa':'ia',
    'Idaho':'id',
    'Indiana':'in',
    'Illinois':'il',
    'Kansas':'ks',
    'Kentucky':'ky',
    'Louisiana':'la',
    'Maine':'me',
    'Massachusetts':'ma',
    'Maryland':'md',
    'Michigan':'mi',
    'Minnesota':'mn',
    'Missouri':'mo',
    'Mississippi':'ms',
    'Montana':'mt',
    'North Carolina':'nc',
    'Nebraska':'ne',
    'New Hampshire':'nh',
    'New Jersey':'nj',
    'New Mexico':'nm',
    'Nevada':'nv',
    'New York':'ny',
    'North Dakota':'nd',
    'Ohio':'oh',
    'Oklahoma':'ok',
    'Oregon':'or',
    'Pennsylvania':'pa',
    'Rhode Island':'ri',
    'South Carolina':'sc',
    'South Dakota':'sd',
    'Tennessee':'tn',
    'Texas':'tx',
    'Utah':'ut',
    'Virginia':'va',
    'Vermont':'vt',
    'Washington':'wa',
    'West Virginia':'wv',
    'Wisconsin':'wi',
    'Wyoming':'wy',
}

#Obtain CFD Image Embeddings
IMG_SOURCE = f'D:\\cfd\\CFD Version 3.0\\divided_images\\N'
image_targets = listdir(IMG_SOURCE)

cfd_list = []

for img in image_targets:

    img_ = Image.open(f'{IMG_SOURCE}/{img}').convert('RGB')
    processed_img = preprocess(img_).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_img(processed_img,apply_preprocess=False).to('cpu').numpy().squeeze()
        cfd_list.append(emb)

cfd_arr = np.array(cfd_list)
cfd_df = pd.DataFrame(cfd_arr,index=image_targets)
cfd_df.to_csv(f'D:\\cfd\\embedding_df_slip.vec',sep=' ')

#Obtain Text Embeddings

#We/They WEAT Stimuli
american_stimuli = ['we','us','our','ours','ourselves','familiar','similar','here']
foreign_stimuli = ['they','them','their','theirs','themselves','other','others','there']

#Defining American Target Stimuli
defining_american_targets = [f'a photo of someone who is patriotic',f'a photo of someone who is an immigrant to America',f'a photo of someone who treats people of all races and backgrounds equally']

#State Target Stimuli
states = list(states_dict.keys())
state_someone = [f'a photo of someone who lives in the state of {state}' for state in states] + [f'a photo of someone who lives in Washington, D.C.']

all_language_targets = american_stimuli + foreign_stimuli + defining_american_targets + state_someone

language_embeddings = []

for target in all_language_targets:

with torch.no_grad():
    emb = model.encode_text([target]).to('cpu').numpy().squeeze()
    language_embeddings.append(emb)

lang_arr = np.array(language_embeddings)
lang_df = pd.DataFrame(lang_arr,index=all_language_targets)
lang_df.to_csv(f'D:\\cfd\\lang_df_slip.vec',sep=' ')

