import os
import numpy as np, argparse, time, pickle, random
import torch

# from model import *
from transformers import AutoModel, AutoTokenizer, pipeline


import warnings
warnings.filterwarnings("ignore")
# We use seed = 100 for reproduction of the results reported in the paper.
# seed = 28
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "1"
tokenizer = AutoTokenizer.from_pretrained("/home/zy/other/CEE/pretrained/roberta-base")

model = AutoModel.from_pretrained("/home/zy/other/CEE/pretrained/roberta-base")
model.eval()

DD_emos = [
    "happily", 
    "neutrally",
    "angrily",
    "sadly",
    "fearfully",
   "surprisingly" ,
    "disgustedly"
]

embeddings = []
feature_extractor = pipeline("feature-extraction",framework="pt",model="/home/zy/other/CEE/pretrained/roberta-base")
embeddings = []
with torch.no_grad():
    for emo in DD_emos:
        # input_ids = tokenizer.encode(emo, add_special_tokens=False, return_tensors='pt')
        # outputs = model(input_ids)
        # last_hidden_state = outputs.last_hidden_state.mean(1)[0]
        emb = feature_extractor(emo,return_tensors = "pt")[0].mean(axis=0)
        # embeddings.append(last_hidden_state)
        embeddings.append(emb.unsqueeze(0))
    embeddings = torch.cat(embeddings, dim=0)
    padding = torch.zeros(1, emb.shape[-1])
    embeddings = torch.cat([embeddings, padding], dim=0)
    torch.save(embeddings, "/home/zy/other/CEE/emo_embeddings/emotion_embeddings.pt")