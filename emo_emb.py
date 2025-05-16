import os
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "1"

save_dir = "./emo_embeddings" 
save_path = os.path.join(save_dir, "emotion_embeddings.pt")


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = AutoModel.from_pretrained("FacebookAI/roberta-base")
model.eval()


DD_emos = [
    "happily", 
    "neutrally",
    "angrily",
    "sadly",
    "fearfully",
    "surprisingly",
    "disgustedly"
]

embeddings = []
feature_extractor = pipeline("feature-extraction", framework="pt", model="FacebookAI/roberta-base")

with torch.no_grad():
    for emo in DD_emos:
        emb = feature_extractor(emo, return_tensors="pt")[0].mean(axis=0)
        embeddings.append(emb.unsqueeze(0))
    embeddings = torch.cat(embeddings, dim=0)
    padding = torch.zeros(1, emb.shape[-1]) 
    embeddings = torch.cat([embeddings, padding], dim=0)
    os.makedirs(save_dir, exist_ok=True)  
    torch.save(embeddings, save_path)
    print(f"Embeddings saved to {save_path}")
