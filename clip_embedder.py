from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
from PIL import Image
import pandas as pd
from os import listdir
import torch
from clip_functions import states_dict

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

#Obtain CFD Image Embeddings
IMG_SOURCE = f'D:\\cfd\\CFD Version 3.0\\divided_images\\N'
image_targets = listdir(IMG_SOURCE)

cfd_list = []

for img in image_targets:

    img_ = Image.open(f'{IMG_SOURCE}\\{img}').convert('RGB')

    with torch.no_grad():
        input_ = processor(images=img_,return_tensors='pt')
        emb = model.get_image_features(**input_).numpy().squeeze()
        cfd_list.append(emb)

cfd_arr = np.array(cfd_list)
cfd_df = pd.DataFrame(cfd_arr,index=image_targets)
cfd_df.to_csv(f'D:\\cfd\\embedding_df_clip_base32.vec',sep=' ')

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
        tokenized_target = tokenizer([target],return_tensors='pt')
        emb = model.get_text_features(**tokenized_target).numpy().squeeze()

    language_embeddings.append(emb)

lang_arr = np.array(language_embeddings)
lang_df = pd.DataFrame(lang_arr,index=all_language_targets)
lang_df.to_csv(f'D:\\cfd\\lang_df_clip_base32.vec',sep=' ')