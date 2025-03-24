import numpy
import pandas as pd
from tqdm import tqdm, trange
import json
import pickle
from transformers import RobertaTokenizer, BertTokenizer, AutoModel, AutoTokenizer, pipeline
import torch
from torch.nn.utils.rnn import pad_sequence
def get_new_input_data(split=None): 
    ori_data = json.load(open('./data/original_annotation/dailydialog_' + split + '.json', 'r', encoding='utf-8'))
    emotion_dict = {'happily': 0, 'neutrally': 1, 'angrily': 2, 'sadly': 3, 'fearfully': 4, 'surprisingly': 5, 'disgustedly': 6}
    
    emotion = {}
    target_context = {}
    speaker = {}
    cause_label = {}
    ids = []
    dialogues = []
    for id, v in tqdm(ori_data.items()):
        conversation = v[0]
        cnt = 1
        for item in conversation:
            turn_data = {}
            if item['emotion'] == 'neutral':
                continue
            turn = item['turn']
            target = item['utterance']
            causal_utterances = item.get('expanded emotion cause evidence', None)
            if causal_utterances == None:
                continue
            cause_labels = [0] * turn
            for index in causal_utterances:
                if index != 'b' and index <= turn:
                    cause_labels[index-1] = 1
            
            context = []
            speaker_list = []
            emo = []
            emo_label = []
            for i in range(turn):
                cur_emo = conversation[i]['emotion']
                # print(cur_emo)
                if cur_emo == 'sadness' or cur_emo == 'sad':
                    cur_emo = 'sadly'
                if cur_emo == 'surprise' or cur_emo == 'surprised':
                    cur_emo = 'surprisingly'
                if cur_emo == 'happiness' or cur_emo == 'excited' or cur_emo == 'happy':
                    cur_emo = 'happily'
                if cur_emo == 'fear':
                    cur_emo = 'fearfully'
                if cur_emo == 'anger' or cur_emo == 'angry':
                    cur_emo = 'angrily'
                if cur_emo == 'neutral':
                    cur_emo = 'neutrally'
                if cur_emo == 'disgust':
                    cur_emo = 'disgustedly'
                # emo.append(emotion_dict[cur_emo])
                emo.append(cur_emo)
                # print(cur_emo)
                emo_label.append(emotion_dict[cur_emo])
                speaker_list.append(conversation[i]['speaker'])
                context.append(conversation[i]['utterance'])
            turn_data["id"] = id
            turn_data["speaker"] = speaker_list
            turn_data["cause"] = cause_labels
            turn_data["emotion"] = emo
            turn_data["text"] = context
            turn_data["emo_label"] = emo_label
            dialogues.append(turn_data)
    return dialogues
    #         ids.append(id + '_' + str(cnt))
    #         target_context[id + '_' + str(cnt)] = context
    #         speaker[id + '_' + str(cnt)] = speaker_list
    #         cause_label[id + '_' + str(cnt)] = cause_labels
    #         emotion[id + '_' + str(cnt)] = emo
    #         cnt += 1
    
    # pickle.dump([target_context, speaker, cause_label, emotion, ids], open('./data/dailydialog_' + split + '.pkl', 'wb'))

def data_statistics(split=None):
    target_context, speaker, cause_label, emotion, ids = pickle.load(open('./data/dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')
    pos_cnt, neg_cnt = 0, 0
    total = cause_label.values()
    for item in total:
        for i in item:
            if i == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1
    print("For " + split + ":")
    print("Pos Pairs:", pos_cnt)
    print("Neg Pairs:", neg_cnt)
    print("Max Length of Conversation:", max([len(item) for item in speaker.values()]))

def label_statistics(split=None):
    target_context, speaker, cause_label, emotion, ids = pickle.load(open('./data/dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')
    pos_cnt, neg_cnt = 0, 0
    total = cause_label.values()
    print(total)
    for item in total:
        for i in item:
            if i == 0:
                neg_cnt += 1
            else:
                pos_cnt += 1


def get_speaker_mask(totalIds, speakers):
    intra_masks, inter_masks = {}, {}
    for i in trange(len(totalIds)):
        id = totalIds[i]
        cur_speaker_list = speakers[id]
        cur_intra_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_inter_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        target_speaker = cur_speaker_list[-1]
        target_index = len(cur_speaker_list) - 1
        
        cur_intra_mask[target_index][target_index] = 1
        for j in range(len(cur_speaker_list) - 1):
            if cur_speaker_list[j] == target_speaker:
                cur_intra_mask[target_index][j] = 1
            else:
                cur_inter_mask[target_index][j] = 1

            # 全连接
            if j == 0:
                cur_intra_mask[j][j] = 1
            else:
                for k in range(j):
                    if cur_speaker_list[j] == target_speaker:
                        cur_intra_mask[j][k] = 1
                    else:
                        cur_inter_mask[j][k] = 1

        intra_masks[id] = cur_intra_mask
        inter_masks[id] = cur_inter_mask
    
    return intra_masks, inter_masks

def get_relative_position(totalIds, speakers):
    relative_position = {}
    thr = 31
    for i in trange(len(totalIds)):
        id = totalIds[i]
        cur_speaker_list = speakers[id]
        cur_relative_position = []
        target_index = len(cur_speaker_list) - 1
        for j in range(len(cur_speaker_list)):
            if target_index - j < thr:
                cur_relative_position.append(target_index - j)
            else:
                cur_relative_position.append(31)
        relative_position[id] = cur_relative_position
    return relative_position

def process_for_ptm(dataset, split, model_size='base'):
    target_context, speaker, cause_label, emotion, ids = pickle.load(open('./data/dailydialog_' + split + '.pkl', 'rb'), encoding='latin1')

    token_ids, attention_mask = {}, {}
    tokenizer = RobertaTokenizer.from_pretrained('./pretrained/roberta-' + model_size) #  "bert-base-uncased"
    
    print("Tokenizing Input Dialogs ...")
    for id, v in tqdm(target_context.items()):
        cur_token_ids, cur_attention_mask = [], []
        for utterance in v:
            encoded_output = tokenizer(utterance)
            tid = encoded_output.input_ids
            atm = encoded_output.attention_mask
            cur_token_ids.append(torch.tensor(tid, dtype=torch.long))
            cur_attention_mask.append(torch.tensor(atm))
        tk_id = pad_sequence(cur_token_ids, batch_first=True, padding_value=1)
        at_mk = pad_sequence(cur_attention_mask, batch_first=True, padding_value=0)
        token_ids[id] = tk_id
        attention_mask[id] = at_mk

    print("Generating Speaker Connections ...")
    intra_mask, inter_mask = get_speaker_mask(ids, speaker)
    
    print("Generating Relative Position ...")
    relative_position = get_relative_position(ids, speaker)
    
    pickle.dump([target_context, token_ids, attention_mask, speaker, cause_label, emotion, relative_position, intra_mask, inter_mask, ids], open('./data/dailydialog_features_roberta_ptm_' + split + '.pkl', 'wb'))

if __name__ == '__main__':
    dataset = 'dailydialog'
    # for split in ['train', 'valid', 'test']:
    #     # get_new_input_data(split)
    #     # process_for_ptm(dataset, split)
    #     data_statistics(split)
    #     label_statistics(split)
    tokenizer = AutoTokenizer.from_pretrained("/home/zy/other/CEE/pretrained/roberta-base")
    emotions = ['sadly', 'surprisingly', 'happily', 'fearfully', 'angrily', 'neutrally', 'disgustedly']
    for emotion in emotions:
        print(tokenizer(emotion)['input_ids'][1:-1])
    feature_extractor = pipeline("feature-extraction",framework="pt",model="/home/zy/other/CEE/pretrained/roberta-base")
    embeddings = []
    
    with torch.no_grad():
        
        # input_ids = tokenizer.encode(emo, add_special_tokens=False, return_tensors='pt')
        # outputs = model(input_ids)
        # last_hidden_state = outputs.last_hidden_state.mean(1)[0]
        emb = feature_extractor(emotions[0], return_tensors = "pt")
        # embeddings.append(last_hidden_state)
        sadly = feature_extractor('Sadly', return_tensors = "pt")[0].mean(axis=0)
    print(emb.shape)
    print(sadly.shape)
    print(torch.cosine_similarity(emb, sadly, dim=0))