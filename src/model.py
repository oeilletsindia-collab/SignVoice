"""Packed BiLSTM encoder and masked additive-attention autoregressive decoder."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self,input_dim=376,hidden=128,dropout=.2):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.ReLU(),nn.Dropout(dropout))
        self.rnn=nn.LSTM(hidden,hidden,batch_first=True,bidirectional=True)
    def forward(self,x,lengths):
        packed=pack_padded_sequence(self.project(x),lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,(h,_)=self.rnn(packed)
        encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=x.size(1))
        mask=torch.arange(x.size(1),device=x.device)[None,:]<lengths.to(x.device)[:,None]
        return encoded,torch.cat([h[-2],h[-1]],dim=-1),mask

class SignTranslator(nn.Module):
    def __init__(self,vocab_size,input_dim=376,hidden=128,embedding=128,dropout=.2):
        super().__init__()
        self.config=dict(vocab_size=vocab_size,input_dim=input_dim,hidden=hidden,embedding=embedding,dropout=dropout)
        self.encoder=Encoder(input_dim,hidden,dropout)
        self.emb=nn.Embedding(vocab_size,embedding,padding_idx=0)
        self.key=nn.Linear(hidden*2,hidden,bias=False)
        self.query=nn.Linear(hidden*2,hidden,bias=False)
        self.score=nn.Linear(hidden,1,bias=False)
        self.cell=nn.GRUCell(embedding+hidden*2,hidden*2)
        self.output=nn.Linear(hidden*4,vocab_size)
        self.dropout=nn.Dropout(dropout)
    def step(self,token,state,encoded,keys,mask):
        scores=self.score(torch.tanh(keys+self.query(state)[:,None,:])).squeeze(-1)
        weights=scores.masked_fill(~mask,float('-inf')).softmax(-1)
        context=torch.bmm(weights[:,None,:],encoded).squeeze(1)
        state=self.cell(torch.cat([self.emb(token),context],-1),state)
        logits=self.output(self.dropout(torch.cat([state,context],-1)))
        return logits,state
    def forward(self,x,lengths,decoder_input):
        enc,state,mask=self.encoder(x,lengths); keys=self.key(enc)
        logits=[]
        for t in range(decoder_input.size(1)):
            prediction,state=self.step(decoder_input[:,t],state,enc,keys,mask)
            logits.append(prediction)
        return torch.stack(logits,1)
    @torch.no_grad()
    def generate(self,x,lengths,max_tokens=64):
        self.eval(); enc,state,mask=self.encoder(x,lengths); keys=self.key(enc)
        token=torch.ones(x.size(0),device=x.device,dtype=torch.long)
        done=torch.zeros_like(token,dtype=torch.bool); generated=[]
        for _ in range(max_tokens):
            logits,state=self.step(token,state,enc,keys,mask)
            logits[:,0:2]=float('-inf')
            token=logits.argmax(-1); token=torch.where(done,torch.full_like(token,2),token)
            generated.append(token); done|=token.eq(2)
            if done.all(): break
        return torch.stack(generated,1)

class SentenceClassifier(nn.Module):
    """Closed-set baseline; its labels are entire repeated training sentences."""
    def __init__(self,num_classes,hidden=128):
        super().__init__(); self.encoder=Encoder(hidden=hidden); self.head=nn.Linear(hidden*2,num_classes)
    def forward(self,x,lengths): return self.head(self.encoder(x,lengths)[1])
