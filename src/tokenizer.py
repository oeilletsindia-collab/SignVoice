import re
from collections import Counter

class Tokenizer:
    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary or ['<pad>','<bos>','<eos>','<unk>']
        if self.vocabulary[:4] != ['<pad>','<bos>','<eos>','<unk>']:
            raise ValueError('Invalid special tokens')
        self.index = {word:i for i,word in enumerate(self.vocabulary)}
    @staticmethod
    def words(text): return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?]",text.lower())
    @classmethod
    def fit(cls,texts,max_vocab=5000):
        count = Counter(w for text in texts for w in cls.words(text))
        return cls(['<pad>','<bos>','<eos>','<unk>']+[w for w,_ in count.most_common(max_vocab-4)])
    def encode(self,text,max_tokens=64):
        return [1]+[self.index.get(w,3) for w in self.words(text)[:max_tokens-2]]+[2]
    def decode(self,ids):
        out=[]
        for i in ids:
            i=int(i)
            if i==2: break
            if i in (0,1): continue
            out.append(self.vocabulary[i] if i<len(self.vocabulary) else '<unk>')
        return re.sub(r'\s+([.,!?])',r'\1',' '.join(out))
