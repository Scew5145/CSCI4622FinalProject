import math
import pickle
import gzip
import numpy as np
import pandas as pd

def getVal(dftrain, id):
    bands, r = [0,1,2,3,4,5], {}
    for b in bands:
        locb = dftrain.loc[(dftrain.iloc[:,0]==id) & (dftrain.iloc[:,2]==b)]
        r[b] = {}
        r[b]['mjd'] = locb.iloc[:,1].values
        r[b]['flux'] = locb.iloc[:,3].values
        r[b]['flux_err'] = locb.iloc[:,4].values
        r[b]['detected'] = locb.iloc[:,5].values
    return r

dftrain = pd.read_csv('./downsample/training10000.csv')
dfmetatrain =  pd.read_csv('./downsample/training_set_metadata.csv')
dftrain.drop(dftrain.columns[0], axis=1, inplace=True)
idcounts = dftrain.iloc[:, 0].value_counts().to_frame('counts')

#downsample
subset_ids = idcounts.iloc[0:5].index     #try max 352 datapoints 
df = pd.DataFrame()
for id in subset_ids:
    df = df.append({'id': id, 'info':getVal(dftrain, id)}, ignore_index=True)

print(df.loc[df.iloc[:,0]==238409]['info'].values[0])
