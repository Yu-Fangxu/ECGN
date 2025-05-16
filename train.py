import numpy as np, argparse, time, pickle, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import DailyDialogRobertaDataset, collate_fn
from loss import MaskedBCELoss2
# from models.emotion_cause_model import KBCIN
from models.model import CLModel
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AdamW, get_constant_schedule, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from transformers import AutoModel, AutoTokenizer
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--bert_path', type=str, default='FacebookAI/roberta-base')
    parser.add_argument('--accumulate_step', type=int, required=False, default=1)
    parser.add_argument('--weight_decay', type=float, required=False, default=3e-4)
    parser.add_argument('--scheduler', type=str, required=False, default='constant')
    parser.add_argument('--use_gpu', type=bool, required=False, default=True)
    parser.add_argument('--batch_size', type=int, default=8, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--num_attention_heads', type=int, default=1, help='Number of output mlp layers.')
    parser.add_argument('--roberta_dim', type=int, default=1024, metavar='HD', help='hidden feature dim')
    parser.add_argument('--seed', type=int, default=42, metavar='seed', help='seed')
    parser.add_argument('--norm', action='store_true', default=False, help='normalization strategy')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save best model')
    parser.add_argument('--use_pos', action='store_true', default=False, help='whether to use position embedding')
    parser.add_argument('--model_size', default='base', help='roberta-base or large')
    parser.add_argument('--model_type', type=str, required=False, default='v2')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
    parser.add_argument('--pad_value', type=int, default=1, help='padding')
    parser.add_argument('--wp', type=int, default=1, help='past window size')
    parser.add_argument('--wf', type=int, default=1, help='future window size')
    parser.add_argument('--n_layers', type=int, default=1, help='num layers of GNN')
    parser.add_argument('--num_bases', type=int, default=2, help='num bases')
    parser.add_argument('--use_bn', type=bool, required=False, default=False)
    args = parser.parse_args()
    
    return args 

def get_paramsgroup(model, warmup=False):
    no_decay = ['bias', 'LayerNorm.weight']
    pre_train_lr = args.lr

    bert_params = list(map(id, model.f_context_encoder.parameters()))
    params = []
    warmup_params = []
    for name, param in model.named_parameters():
        lr = args.lr
        # weight_decay = 0.01
        weight_decay = args.weight_decay
        if id(param) in bert_params:
            lr = pre_train_lr / 4
        if any(nd in name for nd in no_decay):
            weight_decay = 0
        params.append({
            'params': param,
            'lr': lr,
            'weight_decay': weight_decay
        })
        warmup_params.append({
            'params':
            param,
            'lr':
            pre_train_lr / 4 if id(param) in bert_params else lr,
            'weight_decay':
            weight_decay
        })
    if warmup:
        return warmup_params
    params = sorted(params, key=lambda x: x['lr'])
    return params

def get_DailyDialog_loaders(batch_size=8, num_workers=0, pin_memory=False):
    # trainset = DailyDialogRobertaCometDataset('train')
    # validset = DailyDialogRobertaCometDataset('valid')
    # testset = DailyDialogRobertaCometDataset('test')
    bert_path = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    trainset = DailyDialogRobertaDataset('train', tokenizer)
    validset = DailyDialogRobertaDataset('valid', tokenizer)
    testset = DailyDialogRobertaDataset('test', tokenizer)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True, collate_fn=collate_fn)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=collate_fn)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, device, optimizer=None, train=False):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    bert_path = "FacebookAI/roberta-base" 
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    emotions_stat = torch.zeros(7, 7)

    with tqdm(total=int(len(dataloader) / args.accumulate_step), desc=f"Epoch {epoch+1}") as pbar:
        for step, batch in enumerate(dataloader):
            if train:
                optimizer.zero_grad()
            input_ids, label, label_mask, emotions, emo_mask, knowledge_ids, batch_split = batch
            input_ids = input_ids.to(device)
            label = label.to(device)
            label_mask = label_mask.to(device)
            emotions = emotions.to(device)
            emo_mask = emo_mask.to(device)
            knowledge_ids = knowledge_ids.to(device)
            
            if model.training:
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        log_prob, utt_mask = model(input_ids, emotions, emo_mask, knowledge_ids, batch_split, return_mask_output=True) 
                else:
                    log_prob, utt_mask = model(input_ids, emotions, emo_mask, knowledge_ids, batch_split, return_mask_output=True) 
            else:
                if args.fp16:
                    with torch.autocast(device_type="cuda" if args.cuda else "cpu"):
                        with torch.no_grad():
                            log_prob, utt_mask = model(input_ids, emotions, emo_mask, knowledge_ids, batch_split, return_mask_output=True) 
                else:
                    log_prob, utt_mask = model(input_ids, emotions, emo_mask, knowledge_ids, batch_split, return_mask_output=True) 

            # BCE loss
            label_mask = label_mask.eq(1)
            utt_mask = utt_mask.eq(1)


            lp_ = log_prob # [batch, seq_len]
            labels_ = label # [batch, seq_len]
            loss = loss_function(labels_, lp_, label_mask, utt_mask, device)
            if args.accumulate_step > 1:
                loss = loss / args.accumulate_step
            # log_prob = log_prob[utt_mask == 1]
            log_prob = torch.masked_select(lp_, utt_mask)
            labels_ = torch.masked_select(labels_.float(), label_mask)
            
           

            lp_ = log_prob.view(-1)
            labels_ = labels_.view(-1)
            
            pred_ = torch.gt(lp_, 0.5).long() # batch*seq_len
            
            # print(pred_)

            

            preds.append(pred_.cpu().numpy())
            labels.append(labels_.cpu().numpy())
            masks.append(label_mask.view(-1).cpu().numpy())
            losses.append(loss.item()*masks[-1].sum())
            
            if train:
                total_loss = loss
                total_loss.backward()
                
                if (step + 1) % args.accumulate_step == 0:
                    pbar.update(1)
                    optimizer.step()
                step += 1
            else:
                pbar.update(1)

            emos = torch.masked_select(emotions.cpu(), utt_mask)

            end = emos[-1]

            indices = pred_.eq(1).nonzero()

            if sum(indices) == 0:
                continue

            starts = [emos[idx].item() for idx in indices]

            for start in starts:
                emotions_stat[start, end] += 1
        # print("labels:", labels_)
        # print("preds", pred_)
    # print(emotions_stat)
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    dialog = tokenizer.decode(input_ids, skip_special_tokens=True)
    print("preds:", preds)
    print("labels:", labels)
    print("dialog_context:", dialog)


    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_fscore = round(f1_score(labels, preds, average='macro')*100, 2)
    if train == False:
        reports = classification_report(labels,
                                        preds,
                                        target_names=['neg', 'pos'],
                                        # sample_weight=masks,
                                        digits=4)
        return avg_loss, [avg_fscore], reports
    else:
        return avg_loss, [avg_fscore]

def save_badcase(model, dataloader, cuda, args):
    preds, labels = [], []
    scores, vids = [], []
    dialogs = []
    speakers = []
    conv_lens = []

    model.eval()
    dialog_id = 1
    f_out = open('./badcase/badcase_dd.txt', 'w', encoding='utf-8')
    print("Logging Badcase ...")
    for data in tqdm(dataloader):

        r1, r2, r3, r4, x1, x2, x3, x4, x5, x6, o1, o2, o3, \
        qmask, umask, label, emotion_label, relative_position, \
        intra_mask, inter_mask, attention_mask, token_ids, \
        edge_index, edge_type, utterances, speaker, Ids = [data[i].cuda() if i<20 else data[i] for i in range(len(data))] if cuda else data
        attention_mask = [t.cuda() for t in attention_mask]
        token_ids = [t.cuda() for t in token_ids]

        utterances = [u for u in utterances]
        speaker = [s for s in speaker]

        # print(speakers)
        log_prob = model(token_ids, attention_mask, emotion_label, relative_position, edge_index, edge_type, intra_mask, inter_mask, r1, r2, r3, r4, x1, x2, x3, x4, x5, x6, o1, o2, o3, qmask, umask)
        conv_len = torch.sum(umask != 0, dim=-1).cpu().numpy().tolist()
        if model.training:
            log_prob, masked_mapped_output, masked_outputs, proto_scores = model(input_ids, return_mask_output=True) 
            loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
        else:
            with torch.no_grad():
                log_prob, masked_mapped_output, masked_outputs, proto_scores = model(input_ids, return_mask_output=True) 
                loss_output = loss_function(log_prob, masked_mapped_output, label, mask, model)
        # umask = umask.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist() # (B, N)
        pred = torch.gt(log_prob.data, 0.5).long().cpu().numpy().tolist() # (B, N)
        preds += pred
        labels += label
        dialogs += utterances
        speakers += speaker
        conv_len = [item for item in conv_len]
        conv_lens += conv_len

        # finished here

    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels):
            for j in range(conv_lens[i]):
                new_labels.append(label[j])
                new_preds.append(preds[i][j])
    else:
        return

    cases = []
    for i,d in enumerate(dialogs):
        case = []
        for j,u in enumerate(d):
            case.append({
                'text': u,
                'speaker': speakers[i][j],
                'label': labels[i][j],
                'pred': preds[i][j]
            })
            f_out.write(str(dialog_id) + '\t')
            f_out.write(u + '\t')
            f_out.write(speakers[i][j] + '\t')
            f_out.write(str(labels[i][j]))
            f_out.write('\t')
            f_out.write(str(preds[i][j]) + '\n')
        cases.append(case)
        dialog_id += 1

    with open('badcase/dailydialog.json', 'w', encoding='utf-8') as f:
        json.dump(cases,f)

    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    avg_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
    print('badcase saved')
    print('test_f1', avg_fscore)

if __name__ == '__main__':

    args = get_parser()
    print(args)

    # args.cuda = torch.cuda.is_available() and not args.no_cuda
    # args.use_gpu= False
    if args.use_gpu:
        print('Running on GPU')
        args.cuda = True
    else:
        print('Running on CPU')
        args.cuda = False

    # cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    device = "cuda" if args.use_gpu else "cpu"

    fscore_list = []
   
    for seed in [0, 1, 2, 3]: # to reproduce results reported in the paper
        seed_everything(seed)

        model = CLModel(args).to(device)
        # print("model device", model.device)
        # for n, p in model.named_parameters():
        #     if p.requires_grad:
        #         # print(n, p.size())
        #         if len(p.shape) > 1:
        #             torch.nn.init.xavier_uniform_(p)
        #         else:
        #             stdv = 1. / math.sqrt(p.shape[0])
        #             torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        print ('DailyDialog RECCON Model.')

        loss_function = MaskedBCELoss2()

        train_loader, valid_loader, test_loader = get_DailyDialog_loaders(batch_size=batch_size, num_workers=0)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(get_paramsgroup(model.module if hasattr(model, 'module') else model), lr=args.lr)

        scheduler_type = args.scheduler
        if scheduler_type == 'linear':
            num_conversations = len(train_loader.dataset)
            if (num_conversations * n_epochs) % (batch_size * args.accumulate_step) == 0:
                num_training_steps = (num_conversations * n_epochs) / (batch_size * args.accumulate_step)
            else:
                num_training_steps = (num_conversations * n_epochs) // (batch_size * args.accumulate_step) + 1
            num_warmup_steps = int(num_training_steps * args.warmup_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        else:
            scheduler = get_constant_schedule(optimizer)
 
        valid_losses, valid_fscores = [], []
        test_fscores, test_losses, iemocap_test_fscores, iemocap_test_losses = [], [], [], []
        valid_reports, test_reports, iemocap_test_reports = [], [], []
        best_loss, best_label, best_pred, best_mask = None, None, None, None
        best_model = None
    
        max_valid_f1 = 0
        continue_not_increase = 0
        for e in range(n_epochs):
            increase_flag = False
            start_time = time.time()
            train_loss, train_fscore = train_or_eval_model(model, loss_function, train_loader, e, device, optimizer, True)
            valid_loss, valid_fscore, valid_report = train_or_eval_model(model, loss_function, valid_loader,e ,device)
            test_loss, test_fscore, test_report = train_or_eval_model(model, loss_function, test_loader, e, device)
            if valid_fscore[0] > max_valid_f1:
                max_valid_f1 = valid_fscore[0]
                best_model = model
                increase_flag = True
                if args.save:
                    torch.save(model.state_dict(), open('./save_dicts/best_model_{}'.format(str(seed)) + '.pkl', 'wb'))
                    print('Best Model Saved!')
            valid_losses.append(valid_loss)
            valid_fscores.append(valid_fscore)
            valid_reports.append(valid_report)
            test_losses.append(test_loss)
            test_fscores.append(test_fscore)
            test_reports.append(test_report)

            x = 'epoch: {}, train_loss: {}, fscore: {}, valid_loss: {}, fscore: {}, test_loss: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_fscore, valid_loss, valid_fscore, test_loss, test_fscore, round(time.time()-start_time, 2))
            print (x)
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        
        valid_fscores = np.array(valid_fscores).transpose()
        test_fscores = np.array(test_fscores).transpose()
        iemocap_test_fscores = np.array(iemocap_test_fscores).transpose()
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[0][np.argmax(test_fscores[0])]
        report_valid = test_reports[np.argmax(valid_fscores[0])]
        report_test = test_reports[np.argmax(test_fscores[0])]
        scores = [score1, score2]
        scores_val_loss = [score1]
        scores_val_f1 = [score2]
        scores_test_f1 = [score3]
        scores = [str(item) for item in scores]
        print ('Test Scores:')
        print('For RECCON-DD:')
        print('F1@Best Valid Loss: {}'.format(scores_val_loss))
        print('F1@Best Valid F1: {}'.format(scores_val_f1))
        print('F1@Best Test F1: {}'.format(scores_test_f1))
        fscore_list.append(scores_val_f1)
        
        print(report_valid)
        print(report_test)
        
    dd_fscore_mean = np.round(np.mean(fscore_list), 2)
    dd_fscore_std = np.round(np.std(fscore_list), 2)
    log_lines = f'fscore: {dd_fscore_mean}(+-{dd_fscore_std})'
    print(log_lines)
