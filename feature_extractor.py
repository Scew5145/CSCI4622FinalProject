import math
import pickle
import gzip
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pylab as plt
# import seaborn as sns
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

dftrain = pd.read_csv('./all_2/training_set.csv')
dfmetatrain =  pd.read_csv('./all_2/training_set_metadata.csv')
# dftrain.drop(dftrain.columns[0], axis=1, inplace=True)
idcounts = dftrain.iloc[:, 0].value_counts().to_frame('counts')

print(dftrain.head(5))

# len(idcounts)  #7848 total ids

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


def transform(df): 
    df_tf = pd.DataFrame()
    for i in df.id.values: 
        x = df.loc[df.iloc[:,0]==i]['info'].values[0]
        length = min(len(x[0]['mjd']), 
                     len(x[1]['mjd']), 
                     len(x[2]['mjd']), 
                     len(x[3]['mjd']), 
                     len(x[4]['mjd']), 
                     len(x[5]['mjd']))
        for j in range(length):
            b=0
            mjd = j
            b1 = x[b]['flux'][j]
            b2 = x[b+1]['flux'][j]
            b3 = x[b+2]['flux'][j]
            b4 = x[b+3]['flux'][j]
            b5 = x[b+4]['flux'][j]
            b6 = x[b+5]['flux'][j]
            df_tf = df_tf.append({'id': i, 'mjd': mjd, 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5,'b6': b6}, ignore_index=True)

    
    return df_tf



subset_ids = idcounts.index     #try max 352 datapoints 
df = pd.DataFrame()
for id in subset_ids:
    df = df.append({'id': id, 'info':getVal(dftrain, id)}, ignore_index=True)


df_transformed = transform(df)
extraction_settings = ComprehensiveFCParameters()

X = extract_features(df_transformed, 
                     column_id='id', column_sort='mjd',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute)

X.to_csv('X.csv')
y = dfmetatrain['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)

print(classification_report(y_test, cl.predict(X_test)))

y = dfmetatrain['target']
y.index = dfmetatrain['object_id']


#compress 
X_filtered = extract_relevant_features(df_transformed, y, 
                                       column_id='id', column_sort='mjd', 
                                       default_fc_parameters=extraction_settings)
X_filtered.head(5)


X_filtered.to_csv('X_filtered.csv')
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=.1)

cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)

print(classification_report(y_test, cl.predict(X_test)))