import re, numpy as np, pandas as pd, time
from collections import Counter
from gensim.models import FastText
D='/home/claude/dephides/dataset/small_dataset/'
def load(f):
    ys,us=[],[]
    with open(D+f,encoding='utf-8',errors='ignore') as fh:
        for line in fh:
            line=line.rstrip('\n').rstrip('\r')
            if '\t' not in line: continue
            y,u=line.split('\t',1)
            if y in ('legitimate','phishing'): ys.append(int(y=='phishing')); us.append(u.strip())
    return pd.DataFrame({'y':ys,'url':us})
tr,va,te=load('train.txt'),load('val.txt'),load('test.txt')
print(len(tr),len(va),len(te), tr.y.mean(), te.y.mean())
SPLIT=re.compile(r'([.?/\-_=%&@+;:#,])')
def norm(u): return re.sub(r'^[a-z]+://','',u.lower().strip())
def tokseg(u):
    u=norm(u); toks=[t for t in SPLIT.split(u) if t]; seg=[]; s=0
    for t in toks:
        if t=='/' and s==0: s=1
        elif t in('?','#'): s=2
        seg.append(s)
    return toks,seg
def host(u): return re.split(r'[/?#]',norm(u))[0].split(':')[0]
sub=lambda df,n: df.sample(n=min(n,len(df)),random_state=0).reset_index(drop=True)
trs,vas=sub(tr,60000),sub(va,10000); tes=te.reset_index(drop=True)
t0=time.time(); sent=[tokseg(u)[0] for u in tr.url]
ft=FastText(vector_size=100,window=5,min_count=5,min_n=3,max_n=6,workers=2,epochs=3,seed=1)
ft.build_vocab(sent); ft.train(sent,total_examples=len(sent),epochs=ft.epochs)
print('fasttext',time.time()-t0,len(ft.wv.key_to_index))
MAXT,MAXC=66,150
chars=Counter(c for u in trs.url for c in norm(u)); cv={c:i+2 for i,(c,_) in enumerate(chars.most_common(100))}
vocab={'<pad>':0,'<unk>':1}
def enc(df):
    T=np.zeros((len(df),MAXT),np.int32);S=np.full((len(df),MAXT),3,np.int32);C=np.zeros((len(df),MAXC),np.int32)
    for i,u in enumerate(df.url):
        toks,seg=tokseg(u)
        for j,(t,s) in enumerate(zip(toks[:MAXT],seg[:MAXT])):
            if t not in vocab: vocab[t]=len(vocab)
            T[i,j]=vocab[t];S[i,j]=s
        for j,c in enumerate(norm(u)[:MAXC]): C[i,j]=cv.get(c,1)
    return T,S,C
out={}
for k,df in [('tr',trs),('va',vas),('te',tes)]:
    T,S,C=enc(df); out[k+'_T']=T;out[k+'_S']=S;out[k+'_C']=C;out[k+'_y']=df.y.values
E=np.zeros((len(vocab),100),np.float32)
for t,i in vocab.items():
    if i>1: E[i]=ft.wv[t]
out['E']=E
trh=set(host(u) for u in tr.url)
out['te_unseen']=np.array([host(u) not in trh for u in tes.url]); out['nchar']=np.array([len(cv)+2])
print('vocab',len(vocab),'unseen frac',out['te_unseen'].mean(),'phish among unseen',tes.y.values[out['te_unseen']].mean(),'ntest',len(tes))
np.savez_compressed('data.npz',**out)
