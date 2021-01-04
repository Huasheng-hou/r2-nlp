import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam

bert_path = "../bert/"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(768, 10)   # 768 -> 2

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False) # 控制是否输出所有encoder层的结果
        out = self.fc(pooled)   # 得到10分类
        return out


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(DEVICE)
print(model)
