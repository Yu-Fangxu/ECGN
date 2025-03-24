import torch
import torch.nn as nn
from transformers import AutoModel
from sklearn.metrics import euclidean_distances
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_undirected
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, RGATConv, FiLMConv, TransformerConv, GPSConv, GCNConv
from torch.nn import MultiheadAttention
import math 
from torch.autograd import Variable

def pad_features(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

class CLModel(nn.Module):
    def __init__(self, args, tokenizer=None):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.pad_value = args.pad_value
        self.f_context_encoder = AutoModel.from_pretrained(args.bert_path)
        self.eps = 1e-8
        self.mask_value = 0
        if args.model_size == 'base':
            self.dim = 768
        else:
            self.dim = 1024
        # self.rgcn = FiLMConv(self.dim, self.dim // 2, num_relations=3)
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(args.n_layers - 1):
            conv = RGCNConv(self.dim, self.dim, num_relations=1)
            self.convs.append(conv)
            self.relus.append(nn.ReLU())
        
        self.convs.append(RGCNConv(self.dim, self.dim // 2, num_relations=1))
        self.relus.append(nn.ReLU())
        
        self.mlp = nn.Sequential(nn.Linear(2 * self.dim, self.dim), nn.ReLU(), nn.Dropout(args.dropout),
                                 nn.Linear(self.dim, 1))
        emo_embedding = torch.load("./emo_embeddings/emotion_embeddings.pt")
        self.emotion_embedding = nn.Embedding(8, self.dim).from_pretrained(emo_embedding, freeze=False)
        self.dropout = args.dropout
        self.tokenizer = tokenizer
        
        # self.cross_att = CrossAttention(self.dim, heads=args.num_attention_heads, dropout=args.dropout)
        self.transformer = TransformerConv(self.dim // 2, self.dim // 2, heads=self.args.num_attention_heads)
        # self.transformer = GPSConv(channels=self.dim // 2, conv=GCNConv(self.dim // 2, self.dim // 2), heads=self.args.num_attention_heads)
        self.bn = nn.BatchNorm1d(self.dim // 2 * args.num_attention_heads)

        # constant
        self.pi = torch.acos(torch.zeros(1)).item()

    def device(self):
        return self.f_context_encoder.device
    
    def _forward2(self, sentences, emotions, emo_mask, knowledge_ids, batch_split):
        mask = 1 - (sentences == self.pad_value).long()
        mask_know = 1 - (knowledge_ids == self.pad_value).long()
        utterance_encoded = self.f_context_encoder(
            input_ids=sentences,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        
        knowledge_encoded = self.f_context_encoder(
            input_ids=knowledge_ids,
            attention_mask=mask_know,
            output_hidden_states=True,
            return_dict=True
        )['last_hidden_state']
        
        mask_pos = (sentences == self.mask_value).long()

        mask_know_pos = (knowledge_ids == self.mask_value).long()
        # print(mask_know_pos)
        #################### Emotion Embedding ##################
        emotion_encoded = self.emotion_embedding(emotions)
        #################### Emotion Embedding ##################
        
        #################### Modal Fusion ##################
        graph, graph_list = self.batch_graphify(emotion_encoded, utterance_encoded, knowledge_encoded, emo_mask, mask_pos, mask_know_pos, batch_split)
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        for conv, ReLU in zip(self.convs, self.relus):
            x = conv(x=x, edge_index=edge_index, edge_type=edge_attr)
            x = ReLU(x)
            # x = F.dropout(x, p=self.dropout)
        outputs = x
        outputs = self.transformer(x=outputs, edge_index=edge_index)
        if self.args.use_bn:
            outputs = self.bn(outputs)
        
        num_nodes_list = [data.num_nodes for data in graph_list]
        outputs = torch.split(outputs, num_nodes_list)
        #################### Utterance Target ##################
        features = []
        for feature in outputs:
            sent_len = feature.shape[0]
            # sem_feature = feature[:sent_len, :]
            emo_feature = feature
            # g_feature = torch.cat([sem_feature, emo_feature], dim=-1)
            g_feature = emo_feature
            # print(g_feature.shape)
            feature_target = g_feature[-1, :].unsqueeze(0)
            repeat_times = g_feature.shape[0]
            feature_target = feature_target.repeat(repeat_times, 1)
            feature = torch.cat([g_feature, feature_target], dim=-1)
            feature = torch.sigmoid(self.mlp(feature))
            f = torch.zeros(512, 1) - 1
            f[:len(feature)] = feature      
            features.append(f.permute(1, 0))
        # utterance_encoded = torch.cat([utterance_encoded, utt_target], dim=-1)
        # feature = torch.sigmoid(self.mlp(utterance_encoded))
        # print(features[0])
        features = torch.cat(features, dim=0)
        mask_pos = (features != -1).long()
        # print(mask_pos.sum(-1))
        return features, mask_pos
    
    
    def forward(self, sentences, emotions, emo_mask, knowledge_ids, split, return_mask_output=False):
        '''
        generate vector representations for each turn of conversation
        '''
        features, mask_pos = self._forward2(sentences, emotions, emo_mask, knowledge_ids, split)
        
        if return_mask_output:
            return features, mask_pos
        else:
            return features, mask_pos
        
    def batch_graphify(self, emo_sentences, sem_sentences, knowledge_sentences, emo_mask, sem_mask, mask_know_pos, batch_split):
        batch_size = emo_mask.shape[0]
        slen = emo_mask.sum(dim=-1)
        # feature_list, edge_index_list, edge_type_list = [], [], []
        
        data_list = []
        for t in range(batch_size):
            emo_sent = emo_sentences[t]
            # sem_sent = sem_sentences[t]
            # print(emo_mask[i])
            emo_m = emo_mask[t] == 1
            # print(emo_m)
            # sem_m = sem_mask[t] == 1
            # print(sem_m)
            emo_features = emo_sent[emo_m]
            # sem_features = sem_sent[sem_m]
            # print(emo_features.shape)
            # emo_features = self.pe(emo_features.unsqueeze(0)).squeeze(0)    # add position embedding
            # adj_sem = torch.zeros(slen[t], slen[t]).to(self.device())
            adj_emo = torch.zeros(slen[t], slen[t]).to(self.device())
            for i, s in enumerate(range(slen[t])):
                j = max(i - self.args.wp, 0)
                k = min(i + self.args.wf, slen[t])
                # adj_sem[i, j:k] = 1
                adj_emo[i, j:k] = 1
                # print(adj_emo)
                # x_sem = sem_features[i, :]
                # x_sem_target = sem_features[j:k, :]
                # x_sem = x_sem.unsqueeze(0).repeat(x_sem_target.shape[0], 1)
                # sim_sem = F.cosine_similarity(x_sem, x_sem_target, dim=-1).to(self.device())
                # adj_sem[i, j:k] -= torch.acos(sim_sem) / self.pi
                # print(sim_sem) / self.pi
                x_emo = emo_features[i, :]
                x_emo_target = emo_features[j:k, :]
                x_emo = x_emo.unsqueeze(0).repeat(x_emo_target.shape[0], 1)
                sim_emo = F.cosine_similarity(x_emo, x_emo_target, dim=-1).to(self.device())
                adj_emo[i, j:k] -= torch.acos(sim_emo) / self.pi
                
            # edge_index_sem, edge_weight_sem = dense_to_sparse(adj_sem)    # edge_index between semantic embeddings
            edge_index_emo, edge_weight_emo = dense_to_sparse(adj_emo)    # edge_index between emotional embeddings
            # edge_index_sem, edge_weight_sem = to_undirected(edge_index_sem, edge_weight_sem)
            edge_index_emo, edge_weight_emo = to_undirected(edge_index_emo, edge_weight_emo)
            # print("edge_weight:", edge_weight_emo)
            # edge_index_sem = edge_index_sem.to(self.device())
            edge_index_emo = edge_index_emo.to(self.device())
            edge_index_emo += slen[t]                       
            # edge_index_crs = torch.tensor([[i, i + slen[t]] for i in range(0, slen[t])]).T                            # edge_index between semantic and emotional embeddings
            # edge_index_crs = to_undirected(edge_index_crs)
            # edge_index_crs = edge_index_crs.to(self.device())
            # edge_index = torch.cat([edge_index_sem, edge_index_emo, edge_index_crs], dim=-1)
            edge_index = edge_index_emo
            # edge_type_sem = torch.ones(edge_index_sem.shape[-1])
            edge_type_emo = torch.zeros(edge_index_emo.shape[-1])
            # edge_type_crs = torch.ones(edge_index_crs.shape[-1]) + 1
            # edge_type = torch.cat([edge_type_sem, edge_type_emo, edge_type_crs], dim=-1).to(torch.int64)
            edge_weight = torch.ones(edge_index.shape[-1]).to(self.device())
            # edge_weight_crs = torch.ones(edge_index_crs.shape[-1]).to(self.device())
            # edge_weight = torch.cat([edge_weight_sem, edge_weight_crs], dim=-1)
            # features = torch.cat([sem_features, emo_features], dim=0)
            data = Data(x=emo_features, edge_index=edge_index, edge_attr=edge_type_emo, edge_weight=edge_weight)
            data = data.to(self.device())
            # print(data.x.shape)
            data_list.append(data)
        
        batch = Batch.from_data_list(data_list)
        return batch, data_list
    
    def similarity(self, x, y):
        pass
    
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.emo_attention = MultiheadAttention(dim, heads, dropout)
        self.sem_attention = MultiheadAttention(dim, heads, dropout)
        self.q_emo_proj = nn.Linear(dim, dim)
        self.k_emo_proj = nn.Linear(dim, dim)
        self.v_emo_proj = nn.Linear(dim, dim)
        self.q_sem_proj = nn.Linear(dim, dim)
        self.k_sem_proj = nn.Linear(dim, dim)
        self.v_sem_proj = nn.Linear(dim, dim)
        
        self.transformer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        
    def forward(self, emo_sentences, sem_sentences, emo_mask, sem_mask):
        # sem_mask = sem_mask.eq(1) 
        # logits = sem_sentences[sem_mask == 1]
        batch_size = emo_sentences.shape[0]
        outputs = []
        for i in range(batch_size):
            emo_sent = emo_sentences[i]
            sem_sent = sem_sentences[i]
            # print(emo_mask[i])
            emo_m = emo_mask[i] == 1
            # print(emo_m)
            sem_m = sem_mask[i] == 1
            # print(sem_m)
            emo_sent = emo_sent[emo_m]
            sem_sent = sem_sent[sem_m]
            sent_len = emo_sent.shape[0]
            # print("sent_len:", sent_len)
            q_emo = self.q_emo_proj(emo_sent)
            k_emo = self.k_emo_proj(emo_sent)
            v_emo = self.v_emo_proj(emo_sent)
            q_sem = self.q_sem_proj(sem_sent)
            k_sem = self.k_sem_proj(sem_sent)
            v_sem = self.v_sem_proj(sem_sent)
            sem_output, sem_output_weights = self.sem_attention(q_emo, k_sem, v_sem)
            emo_output, emo_output_weights = self.emo_attention(q_sem, k_emo, v_emo)
            # sem_output, sem_output_weights = self.sem_attention(emo_sent, sem_sent, sem_sent)
            # emo_output, emo_output_weights = self.emo_attention(sem_sent, emo_sent, emo_sent)
            output = torch.cat([sem_output, emo_output], dim=0)
            output = self.transformer(output)
            sem_feature = output[:sent_len, :]
            emo_feature = output[sent_len:, :]
            # print(output.shape)
            # print(sem_feature.shape)
            # print(emo_feature.shape)
            output = torch.cat([sem_feature, emo_feature], dim=-1)
            outputs.append(output)
        # q_emo = self.q_emo_proj(emo_sentences)
        # k_emo = self.k_emo_proj(emo_sentences)
        # v_emo = self.v_emo_proj(emo_sentences)
        # q_sem = self.q_sem_proj(sem_sentences)
        # k_sem = self.k_sem_proj(sem_sentences)
        # v_sem = self.v_sem_proj(sem_sentences)
        
        # sem_output, sem_output_weights = self.sem_attention(q_emo, k_sem, v_sem)
        # emo_output, emo_output_weights = self.emo_attention(q_sem, k_emo, v_emo)
        
        # output = torch.cat([sem_output, emo_output], dim=1)
        
        # output = self.transformer(output)
        
        return outputs
        
