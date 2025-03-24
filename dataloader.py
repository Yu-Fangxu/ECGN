import numpy
import torch
from torch.nn.modules import padding
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from get_data import get_new_input_data
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import os
from generate_knowledge import Comet
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pad_matrix(matrix, padding_index=0):
    max_len = max(i.size(0) for i in matrix)
    batch_matrix = []
    for item in matrix:
        item = item.numpy()
        batch_matrix.append(numpy.pad(item, ((0, max_len-len(item)), (0, max_len-len(item))), 'constant', constant_values=(padding_index, padding_index)))
    return batch_matrix


def pad_to_len(list_data, max_len, pad_value):
    list_data = list_data[-max_len:]
    len_to_pad = max_len - len(list_data)
    pads = [pad_value] * len_to_pad
    list_data.extend(pads)
    return list_data

def label_padding(label_list, max_len, value=-1):
    max_length = max_len
    padded_label = []

    for sublist in label_list:
        padded_sublist = sublist + [value] * (max_length - len(sublist))
        padded_label.append(padded_sublist)

    return padded_label


class DailyDialogRobertaDataset(Dataset):
    def __init__(self, split, tokenizer):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        # self.max_len = args.max_len
        self.relations = ["xIntent", "xNeed", "xReact", "xWant"]
        self.max_len = 512
        self.pad_value = 1
        self.tokenizer = tokenizer
        # self.comet = Comet("/home/zy/other/CEE/comet-atomic-2020/comet_atomic2020_bart/comet-atomic_2020_BART")
        # self.comet.model.zero_grad()
        self.knowledge_meta = pickle.load(open('./data/dailydialog_knowledge_' + split + '.pkl', "rb"), encoding='latin1')
        self.knowledge_meta = self.knowledge_meta[split]
        self.data, self.labels, self.emotions, self.speakers, self.knowledge = self.read(split, tokenizer)
        assert len(self.data) == len(self.labels)
        
    def read(self, split, tokenizer):
        dialogs = get_new_input_data(split)
        data_list = []
        label_list = []
        emotion_list = []
        speaker_list = []
        knowledge_list = []
        # knowledge = pickle.load(open('./data/dailydialog_knowledge_' + split + '.pkl', "rb"), encoding='latin1')
        for i, turn_data in tqdm(enumerate(dialogs)):
            utterance_ids = []
            id = turn_data["id"]
            text, know_text = [], []
            speakers = turn_data["speaker"]
            cause_labels = turn_data["cause"]
            emos = turn_data["emotion"]
            context  = turn_data["text"]
            emo_label = turn_data["emo_label"]
            know_text = self.knowledge_meta[i]
            for speaker, con, emo in zip(speakers, context, emos):
                # encode texts
                # t = tokenizer.cls_token + ' ' + speaker + " " + emo + " " + "says: " + con + tokenizer.sep_token
                t = tokenizer.cls_token + ' ' + speaker + " " + "says: " + con + tokenizer.sep_token
                text.append(t)
                # encode knowledge
                # sent = speaker + " " + emo + " " + "says: " + con
                # queries = []
                # for rel in self.relations:
                #     query = "{} {} [GEN]".format(sent, rel)
                #     queries.append(query)
                # rels = self.comet.generate(queries, decode_method="beam", num_generate=1)
                # rels = [(r, content) for r, content in zip(self.relations, rels)]
                
                # know_sent = self.know_to_text(speaker, rels)
                # know_sent = tokenizer(know_sent)['input_ids'][1:-1]
                
                # know_text.append(torch.tensor(know_sent, dtype=torch.long)) # knowledge for each utterance 
            assert len(text) == len(know_text)
            text = ''.join([str(encoded) for encoded in text])
            # know_text = ''.join([str(encoded) for encoded in know_text])
            token_ids = tokenizer(text)['input_ids'][1:-1]
            
            know_text = pad_sequence(know_text, batch_first=True, padding_value=1)
            
            if len(token_ids) > 510:
                token_ids = token_ids[-510:]
                cls_position = token_ids.index(0)
                t_ids = token_ids[cls_position:]
                
                n_utterance = t_ids.count(0)
                emo_label = emo_label[-n_utterance:]
                cause_labels = cause_labels[-n_utterance:]
                speaker = speakers[-n_utterance:]
                know_text = know_text[-n_utterance:]
            data_list.append(token_ids)
            label_list.append(cause_labels)
            emotion_list.append(emo_label)
            speaker_list.append(speaker)
            knowledge_list.append(know_text)
        # utterance_ids = torch.LongTensor(data_list)
        # label_list = label_padding(label_list)
        # label_list = torch.LongTensor(label_list)
        # label_mask = label_list != -1
        return data_list, label_list, emotion_list, speaker_list, knowledge_list

    def know_to_text(self, speaker, relations):
        prompt = self.tokenizer.cls_token + " " + speaker
        for i, rel in enumerate(relations):
            # print(rel[0])
            if rel[0] == "xIntent":
                sent = "intends " + rel[1][0]
            elif rel[0] == "xNeed":
                sent = "needs" + rel[1][0]
            elif rel[0] == "xReact":
                sent = "feels" + rel[1][0]
            # elif rel == "xReason":
            #     sent = " "
            elif rel[0] == "xWant":
                sent = "wants" + rel[1][0]
            prompt += sent
            if i != len(relations) - 1:
                prompt += " and "
            else:
                prompt += self.tokenizer.sep_token
        
        return prompt

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            feature,
            label
            speaker
            length
            text
        '''

        text = self.data[index]

        label = self.labels[index]

        emotion = self.emotions[index]
        
        knowledge = self.knowledge[index]
        
        return text, label, emotion, knowledge
    
    def __len__(self):
        return len(self.data)
    
    # def generate_graph(self, ):
    def stat(self):
        # self.data, self.labels, self.emotions
        print(sum(self.labels))
        
def collate_fn(data):
    token_ids = []
    labels = []
    emotions = []
    knowledge = []
    max_len_know = 0
    for i, d in enumerate(data):
        t_ids = pad_to_len(d[0], 512, 1)
        token_ids.append(t_ids)
        labels.append(d[1])
        # print(d[2])
        emos = pad_to_len(d[2], 512, 7)
        emotions.append(emos)
        # pad knowledge
        knowledge_ids = d[3]
        max_len_know = max(max_len_know, knowledge_ids.shape[-1])
    
    batch_split = []
    
    for i, d in enumerate(data):
        knowledge_ids = d[3]
        knowledge_ids = torch.cat([knowledge_ids, torch.ones(knowledge_ids.shape[0], max_len_know - knowledge_ids.shape[1], dtype=torch.long)], dim=1)
        # print(knowledge_ids.shape)
        knowledge.append(knowledge_ids)
        batch_split.append(knowledge_ids.shape[0])

    batch_split = torch.LongTensor(batch_split)
    knowledge_ids = torch.cat(knowledge, dim=0)
    
    labels = label_padding(labels, 512, -1)
    token_ids = torch.LongTensor(token_ids)
    labels = torch.LongTensor(labels)
    emotions = torch.LongTensor(emotions)
    knowledge_ids = torch.LongTensor(knowledge_ids)
    label_mask = (labels != -1).long()
    emo_mask = (emotions != 7).long()
    return token_ids, labels, label_mask, emotions, emo_mask, knowledge_ids, batch_split

    
if __name__ == "__main__":
    bert_path = "/home/zy/other/CEE/pretrained/roberta-" + args.model_size
    tokenizer = AutoTokenizer.from_pretrained("/home/zy/other/CEE/pretrained/roberta-"+ args.model_size)
    data = DailyDialogRobertaDataset(split='test', tokenizer=tokenizer)
    print(data[-1])
    loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn)
    for batch in loader:
        print(batch)
        # s = 1
        break