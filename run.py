import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
d=np.load('data.npz'); E=d['E']; NCH=int(d['nchar'][0]); V=E.shape[0]; MAXT,MAXC=66,150
class AttnPool(L.Layer):
    def __init__(s,units=64,**k): super().__init__(**k); s.units=units
    def build(s,shape):
        s.W=s.add_weight(shape=(shape[-1],s.units),initializer='glorot_uniform',name='W'); s.b=s.add_weight(shape=(s.units,),initializer='zeros',name='b'); s.v=s.add_weight(shape=(s.units,1),initializer='glorot_uniform',name='v')
    def call(s,h,mask):
        u=tf.tanh(tf.tensordot(h,s.W,1)+s.b); e=tf.squeeze(tf.tensordot(u,s.v,1),-1); e=tf.where(mask>0,e,-1e9); a=tf.nn.softmax(e,axis=-1)
        return tf.reduce_sum(h*a[...,None],axis=1)
class MaskMax(L.Layer):
    def call(s,h,mask): return tf.reduce_max(tf.where(mask[...,None]>0,h,-1e9),axis=1)
class Gate(L.Layer):
    def __init__(s,units=128,**k): super().__init__(**k); s.units=units
    def build(s,shapes): s.Wa=L.Dense(s.units,activation='tanh'); s.Wb=L.Dense(s.units,activation='tanh'); s.Wg=L.Dense(s.units,activation='sigmoid')
    def call(s,a,b): g=s.Wg(tf.concat([a,b],-1)); return g*s.Wa(a)+(1-g)*s.Wb(b)
def build(name):
    tok=keras.Input((MAXT,),dtype='int32',name='tok'); seg=keras.Input((MAXT,),dtype='int32',name='seg'); chr_=keras.Input((MAXC,),dtype='int32',name='chr')
    mask=L.Lambda(lambda t: tf.cast(t>0,tf.float32))(tok)
    emb=L.Embedding(V,100,weights=[E],trainable=False)(tok)
    def tokview(use_seg,pool):
        x=emb
        if use_seg: x=L.Add()([emb,L.Embedding(4,100,embeddings_initializer=keras.initializers.RandomNormal(0,0.05))(seg)])
        x=L.Dropout(0.3)(x); h=L.Bidirectional(L.GRU(64,return_sequences=True))(x)
        return AttnPool(64)(h,mask) if pool=='attn' else MaskMax()(h,mask)
    def charview():
        c=L.Dropout(0.2)(L.Embedding(NCH,32)(chr_))
        p=[L.GlobalMaxPooling1D()(L.Conv1D(64,k,padding='same',activation='relu')(c)) for k in (3,5,7)]
        return L.Dense(128,activation='relu')(L.Concatenate()(p))
    if name=='cnn':
        x=L.Dropout(0.3)(emb)
        for f in (64,64): x=L.MaxPooling1D(2)(L.Conv1D(f,4,activation='relu')(x))
        z=L.Dense(64,activation='relu')(L.GlobalMaxPooling1D()(L.Conv1D(128,4,activation='relu')(x)))
    elif name=='bilstm':
        x=L.Dropout(0.3)(emb); x=L.Bidirectional(L.LSTM(64,return_sequences=True))(x); x=L.Bidirectional(L.LSTM(32,return_sequences=True))(x); x=L.Bidirectional(L.LSTM(16))(x); z=L.Dense(64,activation='relu')(x)
    elif name=='bigru':
        x=L.Dropout(0.3)(emb); x=L.Bidirectional(L.GRU(64,return_sequences=True))(x); x=L.Bidirectional(L.GRU(32))(x); z=L.Dense(64,activation='relu')(x)
    elif name=='cnn_bilstm':
        x=L.Dropout(0.3)(emb); x=L.MaxPooling1D(3)(L.BatchNormalization()(L.Conv1D(128,4,activation='relu')(x))); z=L.Dense(64,activation='relu')(L.Bidirectional(L.LSTM(100))(L.Dropout(0.3)(x)))
    elif name=='sadg': z=Gate(128)(tokview(True,'attn'),charview())
    elif name=='no_seg': z=Gate(128)(tokview(False,'attn'),charview())
    elif name=='no_char': z=L.Dense(128,activation='relu')(tokview(True,'attn'))
    elif name=='char_only': z=charview()
    elif name=='concat': z=L.Dense(128,activation='relu')(L.Concatenate()([tokview(True,'attn'),charview()]))
    elif name=='maxpool': z=Gate(128)(tokview(True,'max'),charview())
    out=L.Dense(1,activation='sigmoid')(L.Dropout(0.3)(z))
    m=keras.Model([tok,seg,chr_],out); m.compile(keras.optimizers.Adam(1e-3),'binary_crossentropy',metrics=['accuracy']); return m
X=lambda k:{'tok':d[k+'_T'],'seg':d[k+'_S'],'chr':d[k+'_C']}
def metrics(y,p):
    yh=(p>=0.5).astype(int); return dict(acc=accuracy_score(y,yh),prec=precision_score(y,yh,zero_division=0),rec=recall_score(y,yh),f1=f1_score(y,yh),mcc=matthews_corrcoef(y,yh),auc=roc_auc_score(y,p))
if __name__=='__main__':
    name,seed=sys.argv[1],int(sys.argv[2]); EP=int(sys.argv[3]) if len(sys.argv)>3 else 4
    keras.utils.set_random_seed(seed); m=build(name); t0=time.time()
    h=m.fit(X('tr'),d['tr_y'],validation_data=(X('va'),d['va_y']),batch_size=256,epochs=EP,verbose=0,callbacks=[keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True,monitor='val_loss')])
    tt=time.time()-t0; p=m.predict(X('te'),batch_size=512,verbose=0).ravel(); y=d['te_y']; u=d['te_unseen']
    res=dict(model=name,seed=seed,params=int(m.count_params()),train_s=tt,epochs=len(h.history['loss']),hist_val_acc=[float(v) for v in h.history['val_accuracy']],hist_val_loss=[float(v) for v in h.history['val_loss']],hist_tr_loss=[float(v) for v in h.history['loss']],test=metrics(y,p),unseen=metrics(y[u],p[u]),seen=metrics(y[~u],p[~u]))
    idx=np.random.RandomState(0).choice(len(y),100,replace=False); xs=X('te'); m.predict({k:v[:1] for k,v in xs.items()},verbose=0); t1=time.time()
    for i in idx: m.predict({k:v[i:i+1] for k,v in xs.items()},verbose=0)
    res['lat_ms']=(time.time()-t1)/100*1000
    np.save(f'pred_{name}_{seed}.npy',p)
    with open('results.jsonl','a') as f: f.write(json.dumps(res)+'\n')
    os.makedirs('/mnt/user-data/outputs/_backup',exist_ok=True); os.system('cp results.jsonl /mnt/user-data/outputs/_backup/')
    print(name,seed,res['test'],'train_s',round(tt),'lat',round(res['lat_ms'],1),flush=True)
