import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


class Model(nn.Module):
    def __init__(self, cls):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(768, cls)  # 768 -> 2

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=True)  # 控制是否输出所有encoder层的结果
        out = self.fc(pooled)  # 得到10分类
        return [out]


class R2Net(nn.Module):
    def __init__(self, cls):
        super(R2Net, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # FINE TUNE
        self.alpha = torch.nn.Parameter(torch.FloatTensor(12), requires_grad=True)
        self.alpha.data.fill_(1.0 / 12)

        self.K = 3
        self.CNNS = nn.ModuleList()
        for kernel in range(1, self.K + 1):
            self.CNNS.append(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel, 1)))

        self.linear_l = nn.Linear(self.K*768*2, 768)

        self.mlp1 = nn.Sequential(
            nn.Linear(768*2, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, cls)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(768, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, cls),
            nn.Softmax()
        )

        self.fc = nn.Linear(768*2, cls)  # 768 -> 2

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        h, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=True)  # 控制是否输出所有encoder层的结果

        ''' 取最后一层的[CLS] token对应的encoder向量作为global encoding '''
        v_g = h[11][:, 0, :]

        batch_size = context.size()[0]
        sen_len = context.size()[1]

        DEVICE = h[0].device

        ''' 对各层的 encoding vector 进行加权求和 '''
        H = torch.zeros(batch_size, sen_len-1, 768).to(DEVICE)

        for l_index in range(12):
            H += self.alpha[l_index] * h[l_index][:, 1:, :]

        H = H.unsqueeze(1)
        h = []

        for k in range(self.K):
            H_k = self.CNNS[k](H)
            h_max = F.max_pool2d(H_k, kernel_size=(47-k, 1))
            h_avg = F.avg_pool2d(H_k, kernel_size=(47-k, 1))
            h.append(h_max)
            h.append(h_avg)

        ''' 得到 local encoding '''
        h = torch.cat(h, dim=3)
        v_l = self.linear_l(h).squeeze(dim=1).squeeze(dim=1)
        v_l = F.relu(v_l)

        v = torch.cat([v_g, v_l], dim=1)

        # out = self.mlp1(pooled)
        # out = F.softmax(out)

        out = self.fc(v)  # 得到10分类
        return [out]


class LEM(nn.Module):
    def __init__(self, cls):
        super(LEM, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(768, cls)  # 768 -> 2

        ''' number of classes '''
        self.cls = cls

        ''' dimension of joint embedding space '''
        self.embed_dimension = 768

        ''' 随机初始化 label embedding '''
        self.label_embeddings = torch.nn.Parameter(torch.FloatTensor(torch.Size([self.cls, self.embed_dimension])),
                                                   requires_grad=True)
        torch.nn.init.xavier_normal_(self.label_embeddings)

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=True)  # 控制是否输出所有encoder层的结果

        batch_size = context.size()[0]
        sen_len = context.size()[1]

        ''' V is word embedding '''
        V = _[11]

        ''' m is weighted coefficients '''
        m = []

        ''' 对于每一个token（word） '''
        for idx in range(sen_len):
            u_l = []

            ''' 计算和每个label embedding vector之间的cosine similarity，提取相似度最高的结果作为该token的权重系数 '''
            for c in range(self.cls):
                u_l.append(torch.cosine_similarity(V[:, idx, :], self.label_embeddings[c, :], dim=-1))
            u_l = torch.stack(u_l)
            u_l = u_l.max(dim=0).values
            m.append(u_l)

        ''' 对系数矩阵进行softmax '''
        m = torch.stack(m).t()
        beta = F.softmax(m, dim=1)

        z = []

        for idx in range(batch_size):
            z.append(torch.matmul(V[idx, :, :].t(), beta[idx]))

        z = torch.stack(z)

        out = self.fc(z)  # 得到10分类
        if self.training:
            return [out]
        else:
            return [out, z, self.label_embeddings]
