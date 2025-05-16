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

def get_DailyDialog_loaders(tokenizer, batch_size=8, num_workers=0, pin_memory=False):
    trainset = DailyDialogRobertaDataset('train', tokenizer)
    validset = DailyDialogRobertaDataset('valid', tokenizer)
    testset = DailyDialogRobertaDataset('test', tokenizer)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True, collate_fn=collate_fn)

    valid_loader = DataLoader(validset,
                              batch_size=1,
                              num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=collate_fn)

    test_loader = DataLoader(testset,
                             batch_size=1,
                             num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, device, bert_path, optimizer=None, mode="train"):
    label_index_mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    if mode == "train":
        model.train()
    else:
        model.eval()

    emotions_stat = torch.zeros(7, 7)
    error_stats = {}
    correct_count = []

    for step, batch in enumerate(dataloader):
        if mode == "train":
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
        emo_mask = emo_mask.eq(1).cpu()
        lp_ = log_prob # [batch, seq_len]
        labels_ = label # [batch, seq_len]
        if mode == "train":
            loss = loss_function(labels_, lp_, label_mask, utt_mask, device)
        else:
            loss = torch.tensor([0])
        if args.accumulate_step > 1:
            loss = loss / args.accumulate_step
        log_prob = torch.masked_select(lp_, utt_mask)
        labels_ = torch.masked_select(labels_.float(), label_mask)
        emos = torch.masked_select(emotions.cpu(), utt_mask)
        if mode == "test":
            input_ids_list = input_ids.tolist()
            emos_check = emos.view(len(input_ids_list), -1)
            pred_check = torch.gt(log_prob, 0.5).long().view(len(input_ids_list), -1)
            labels_check = labels_.view(len(input_ids_list), -1)
            for index, input_ids in enumerate(input_ids_list):
                dialog = tokenizer.decode(input_ids, skip_special_tokens=True)
                error = pred_check[index].cpu() != labels_check[index].cpu()
                if len(emos) > 1:
                    if emos[-1] != emos[-2]:
                        if error[-2] == True:
                            correct_count.append(0)
                        else:
                            correct_count.append(1)
        lp_ = log_prob.view(-1)
        labels_ = labels_.view(-1)
        pred_ = torch.gt(lp_, 0.5).long() # batch*seq_len
        preds.append(pred_.cpu().numpy())
        labels.append(labels_.cpu().numpy())
        masks.append(label_mask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
        if mode == "train":
            total_loss = loss
            total_loss.backward()
            if (step + 1) % args.accumulate_step == 0:

                optimizer.step()
            step += 1
        else:
            pass
        end = emos[-1]
        indices = pred_.eq(1).nonzero()
        if sum(indices) == 0:
            continue
        starts = [emos[idx].item() for idx in indices]
        for start in starts:
            emotions_stat[start, end] += 1
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_fscore = round(f1_score(labels, preds, average='macro')*100, 2)
    if mode == "valid" or mode == "test":
        if mode == "test":
            print("correct_ratio:", sum(correct_count)/len(correct_count))
        reports = classification_report(labels,
                                        preds,
                                        target_names=['neg', 'pos'],
                                        # sample_weight=masks,
                                        digits=4)
        return avg_loss, [avg_fscore], reports
    else:
        return avg_loss, [avg_fscore]

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
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    for seed in [1, 2, 3, 4]: # to reproduce results reported in the paper
        seed_everything(seed)

        model = CLModel(args, tokenizer).to(device)
        print ('DailyDialog RECCON Model.')

        loss_function = MaskedBCELoss2()
        
        train_loader, valid_loader, test_loader = get_DailyDialog_loaders(tokenizer=tokenizer, batch_size=batch_size, num_workers=0)
        
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
            train_loss, train_fscore = train_or_eval_model(model, loss_function, train_loader, e, device, args.bert_path, optimizer, mode="train")
            valid_loss, valid_fscore, valid_report = train_or_eval_model(model, loss_function, valid_loader,e ,device, args.bert_path, mode="valid")
            test_loss, test_fscore, test_report = train_or_eval_model(model, loss_function, test_loader, e, device, args.bert_path, mode="test")
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
