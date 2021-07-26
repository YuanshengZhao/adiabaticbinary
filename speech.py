
import numpy as np
import os
import scipy.io.wavfile as sp_wr
import random

random.seed(0)

def load_data():

    speech_path=os.path.join('speech-command/')

    path =[speech_path+"zero/",
           speech_path+"one/",
           speech_path+"two/",
           speech_path+"three/",
           speech_path+"four/",
           speech_path+"five/",
           speech_path+"six/",
           speech_path+"seven/",
           speech_path+"eight/",
           speech_path+"nine/"]

    ncls=len(path)

    fname1=[os.listdir(ii) for ii in path]

    fname=[[] for _ in path]

    for i in range(ncls):
        ll=len(fname1[i])
        j=0
        while(j<ll):
            k=j+1
            while(k<ll and fname1[i][j][:8]==fname1[i][k][:8]):
                k+=1
            fname[i]+=random.sample(fname1[i][j:k], 1)
            j=k

    ntrain=1200
    nval=100
    ntest=100
    nfile=ntrain+nval+ntest
    print([len(ii) for ii in fname])

    fname_s=[random.sample(ii, nfile) for ii in fname]
    for tg in range(ncls):
        for i in range(nfile):
            fname_s[tg][i]=[tg, path[tg]+fname_s[tg][i]]
    trn_ff=[]
    for tg in range(ncls):
        trn_ff+=fname_s[tg]
    random.shuffle(trn_ff)

    x_t=[(sp_wr.read(ii[1])[1])/1000.0 for ii in trn_ff]
    y_t=np.array([ii[0] for ii in trn_ff])

    lgstd,dspl = 16000,2
    for ii in range(ncls*nfile):
        lg=len(x_t[ii])
        if(lg<lgstd):
            x_t[ii]=np.concatenate([x_t[ii]-np.mean(x_t[ii]),np.zeros(lgstd-lg)])
        elif(lg==lgstd):
            x_t[ii]=x_t[ii]-np.mean(x_t[ii])
        else:
            print("Warning: wav tool long!")
        x_t[ii]=x_t[ii][np.arange(0,lgstd,dspl)]  #downsample
        stdd=np.std(x_t[ii])
        if(stdd<1e-3):
            print("Warning: small std!")
        x_t[ii]=x_t[ii]/stdd
    x_t=np.vstack(x_t)


    lgstd=lgstd//dspl

    x_train_t=(x_t[:ncls*ntrain]).reshape([ncls*ntrain,lgstd,1])
    x_val_t=(x_t[ncls*ntrain:ncls*(ntrain+nval)]).reshape([ncls*nval,lgstd,1])
    x_test_t=(x_t[ncls*(ntrain+nval):]).reshape([ncls*ntest,lgstd,1])

    y_train_t=(y_t[:ncls*ntrain])
    y_val_t=(y_t[ncls*ntrain:ncls*(ntrain+nval)])
    y_test_t=(y_t[ncls*(ntrain+nval):])

    return (x_train_t,y_train_t),(x_val_t,y_val_t),(x_test_t,y_test_t)

(x_train,y_train),(x_val,y_val),(x_test,y_test) = load_data()

np.savez("Dspeech-command/numbers.npz",x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val,x_test=x_test,y_test=y_test)