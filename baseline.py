import math
import pickle
import gzip
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


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
subset_ids = idcounts.iloc[0:2].index     #try max 352 datapoints 
df = pd.DataFrame()
for id in subset_ids:
    df = df.append({'id': id, 'info':getVal(dftrain, id)}, ignore_index=True)

# x = df.loc[df.iloc[:,0]==238409]['info'].values
# print(x[0][1].keys())    # [which datapoint in the subset][band]
# print(x[0][2]['flux'])

pyplot.figure()
for id in subset_ids:
    x = df.loc[df.iloc[:,0]==id]['info'].values
    for i in [0,1,2,3,4,5]:
        pyplot.subplot(6, 1, i+1)
        pyplot.plot(x[0][i]['flux'])
        pyplot.title(i, y=0.5, loc='right')
pyplot.show()


groupcnts = np.sum(dfmetatrain.iloc[:, 11].value_counts().to_frame('counts'))
avgcnts = groupcnts/14
print(avgcnts)


######bootstrapping
def balance(df):     
    return resample(df, 
                    replace=True, 
                    n_samples=2300,   #max target 90
                    random_state=123)

dfmetatrain =  pd.read_csv('./downsample/training_set_metadata.csv')
idcounts = dfmetatrain.iloc[:, 11].value_counts().to_frame('counts')
# print(idcounts)
df90 = dfmetatrain.loc[dfmetatrain.iloc[:,11]==90]
minority_class = [42, 65, 16, 15, 62, 88, 92, 67, 52, 95, 6, 64, 53]
upsampled_list = [df90]                    
for m in minority_class:
    df = dfmetatrain.loc[dfmetatrain.iloc[:,11]==m]
    upsampled_list.append(balance(df))
dfmetatrain_upsampled = pd.concat([i for i in upsampled_list])
dfmetatrain_upsampled.drop(['distmod'], axis=1, inplace=True)
# dfmetatrain_upsampled['target'].value_counts()




#Baseline with multinomial logistic regression

X, y = dfmetatrain_upsampled.iloc[:, 1:10], dfmetatrain_upsampled.iloc[:, 10]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    shuffle=True,
                                                    random_state=123)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
print("Splitting ratio 10%")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# print(np.unique(y_train))
normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)

pca = PCA(n_components=6)    
X_train, X_test = pca.fit_transform(X_train), pca.fit_transform(X_test)

logreg = LogisticRegression(C=10.0, penalty="l2",solver='newton-cg', multi_class='multinomial')
logreg = LogisticRegression()
logreg.fit(X_train, y_train) 
score = cross_val_score(logreg, X_train, y_train, cv = 10, n_jobs = 2) 
print(score)
ypred = logreg.predict(X_test)
print("Baseline accuracy score: "+str(accuracy_score(y_test, ypred)))   # ~= 1/14
