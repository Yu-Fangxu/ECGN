import torch
import torch.nn as nn

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, label, logits, label_mask, utt_mask):
        # label: [batch_size, conv_len, conv_len]
        # logits: [batch_size, conv_len, conv_len]
        # mask: [batch_size, conv_len, conv_len]
        label_mask = label_mask.eq(1)
        label = torch.masked_select(label.float(), label_mask)
        utt_mask = utt_mask.eq(1) 
        logits = logits[utt_mask == 1]
        loss = self.loss_fn(logits.squeeze(), label.float())
        return loss
    
class MaskedBCELoss2(nn.Module):
    def __init__(self):
        super(MaskedBCELoss2, self).__init__()
        self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, label, logits, label_mask, utt_mask, device):
        # label: [batch_size, conv_len, conv_len]
        # logits: [batch_size, conv_len, conv_len]
        # mask: [batch_size, conv_len, conv_len]
        # print(logits.shape)
        # print(label.shape)
        label_mask = label_mask.eq(1)
        label = torch.masked_select(label.float(), label_mask)
        utt_mask = utt_mask.eq(1) 
        logits = torch.masked_select(logits, utt_mask)
        logits = logits.to(device)
        label = label.to(device)
        loss = self.loss_fn(logits.squeeze(), label.float())
        return loss