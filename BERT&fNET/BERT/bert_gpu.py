
import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
word_list = list(set(" ".join(sentences).split()))
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)

# model parameters
maxlen = 30
batch_size = 6
max_pred = 5
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 4 * d_model
d_k = d_v = 64 # dimension of K(=Q), V
n_segments = 2
model_save_path = './model.ckpt'


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_data():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        n_pred = min(max_pred, min(1, int(len(input_ids) * 0.15)))
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.9:
                index = randint(0, vocab_size - 1)
                while index < 4:
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index

        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 池化 zero padding
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch


batch = make_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids = torch.LongTensor(input_ids)
segment_ids, masked_tokens = torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens)
masked_pos, isNext = torch.LongTensor(masked_pos), torch.LongTensor(isNext)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids,  segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)


# 模型构建
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # ????
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        pos = pos.to(device)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.output_layer = nn.Linear(n_heads * d_v, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residule, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.output_layer(context)
        return self.layernorm(output + residule)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoerLayer(nn.Module):
    def __init__(self):
        super(EncoerLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoerLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
        h_pooled = self.fc(output[:, 0])
        logits_clsf = self.classifier(h_pooled)  ## 是否是下一句
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.activ2(self.linear(h_masked))
        logits_lm = self.fc2(h_masked)   ## mask位置词预测
        return logits_lm, logits_clsf


model = BERT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)


# 训练
torch.cuda.empty_cache()
for epoch in range(180):
    min_loss = 1e9
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        # print(input_ids.size(0))
        input_ids, segment_ids, masked_pos = input_ids.to(device), segment_ids.to(device), masked_pos.to(device)
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext.to(device)) # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 模型保存
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), model_save_path)

# # 测试
# # Predict mask tokens ans isNext
# input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
# print(text)
# print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])
#
# logits_lm, logits_clsf = model(torch.LongTensor([input_ids]),
#                  torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
# logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
# print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
# print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])
#
# logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
# print('isNext : ', True if isNext else False)
# print('predict isNext : ',True if logits_clsf else False)