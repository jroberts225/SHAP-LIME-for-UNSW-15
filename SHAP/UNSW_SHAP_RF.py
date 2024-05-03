print('--------------------------------------------------')
print('RF sensor and shap with Lime')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Makes sure we see all columns

from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.datasets import load_iris
# Loading Scikits random fo
#rest classifier
import sklearn
import lime
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
#from sklearn.metrics import auc_score
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.preprocessing import label_binarize
#loading pandas
import pandas as pd
#Loading numpy
import numpy as np
# Setting random seed
import time

import shap
from scipy.special import softmax
np.random.seed(0)
import matplotlib.pyplot as plt
import random

from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', None)

print('Defining Function')
print('--------------------------------------------------')

def oversample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    # Convert to numpy and oversample
    x_np = X_train.to_numpy()
    y_np = y_train.to_numpy()
    x_np, y_np = oversample.fit_resample(x_np, y_np)
    
    # Convert back to pandas
    x_over = pd.DataFrame(x_np, columns=X_train.columns)
    y_over = pd.Series(y_np)
    return x_over, y_over
def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the feature
s, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

def ACC(x,y,w,z):
    Acc = (x+y)/(x+w+z+y)
    return Acc

def PRECISION(x,w):
    Precision = x/(x+w)
    return Precision
def RECALL(x,z):
    Recall = x/(x+z)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(x,y,w,z):
    BACC =(x/(x+z)+ y/(y+w))*0.5
    return BACC
def MCC(x,y,w,z):
    MCC = (y*x-z*w)/(((x+w)*(x+z)*(y+w)*(y+z))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
     # plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
      #print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
      auc_avg += auc(fpr[i], tpr[i])
      counting = i+1
    return auc_avg/counting

print('Selecting Column Features')
print('--------------------------------------------------')

req_cols = ['id', 'dur', 'proto', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smean', 'dmean', 'trans_depth', 'sjit', 'djit', 'sinpkt', 'dinpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']

print('Loading Database')
print('--------------------------------------------------')

df0 = pd.read_csv ('./UNSW_NB15_training-set.csv', usecols=req_cols)

df1 = pd.read_csv ('./UNSW_NB15_testing-set.csv', usecols=req_cols)


frames = [df0, df1]
df = pd.concat(frames,ignore_index=True)
df = df.sample(frac=0.35,replace=True,random_state=1)

le = LabelEncoder()
label = le.fit_transform(df['proto'])
label2 = le.fit_transform(df['attack_cat'])
df.drop("proto",axis=1,inplace=True)
df["proto"] = label

print('---------------------------------------------------------------------------------')
print('Separating features and labels')
print('---------------------------------------------------------------------------------')
print('')
y = df.pop('attack_cat')
X = df
# summarize class distribution
counter = Counter(y)
# transform the dataset
print('---------------------------------------------------------------------------------')
result_list = [counter['None'],counter['Analysis'], counter['Generic']]
print('number of Labels  ',result_list)
print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')
df_max_scaled = df.copy()
df_max_scaled
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign( Label = y)
df = df.fillna(0)

print('---------------------------------------------------------------------------------')
print('Counting labels')
print('---------------------------------------------------------------------------------')
print('')
y = df.pop('Label')
X = df
# summarize class distribution
counter = Counter(y)
print(counter)
df = X.assign( Label = y)
df = df.drop_duplicates()
y = df.pop('Label')
X = df
# summarize class distribution
counter = Counter(y)
print('after removing duplicates:',counter)

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, y, train_size=0.80)
df = X.assign( Label = y)

print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')
counter = Counter(labels_train)
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        train, labels_train = oversample(train, labels_train)
counter = Counter(labels_train)

df = pd.get_dummies(df,columns=['id', 'dur', 'proto', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'Label'])

print('Defining the model')
print('--------------------------------------------------')
rf = RandomForestClassifier(max_depth = 5,  n_estimators = 10, min_samples_split = 2, n_jobs = -1,random_state = 1)
#------------------------------------------------------------------------------

print('Training the model')
print('------------------------------------------------------------------------------')
#START TIMER MODEL
start = time.time()
model = rf.fit(train, labels_train)
#END TIMER MODEL
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')

print('------------------------------------------------------------------------------')
#------------------------------------------------------------------------------

predict = rf.predict(test)
labels_test = labels_test.to_numpy()
labels_test = pd.Series(labels_test)
predict = pd.Series(predict)
u, label = pd.factorize(df['label']) 
test2 = test
test = test.to_numpy()

# extracting label name
notused, y_labels = pd.factorize(y)
# Transforming numpy format list
y_labels = list(y_labels)

# Creating, and generating shap values
explainer = shap.TreeExplainer(rf)
start_index = 0
end_index = 1000
shap_values = explainer.shap_values(test[start_index:end_index])
shap_obj = explainer(test[start_index:end_index])

row = 30 # datapoint to explain
prediction = rf.predict(test[row:row+1])[0] # Prediction of the sample
print(f"The RF predicted: {prediction}")
print("------------------------------------------")
print( 'Actual value', labels_test[row:row+1])
print("------------------------------------------")
#extract the index accordingly to prediction
index = y_labels.index(prediction)
#generating shap values explainer
sv = explainer(test[start_index:end_index]) 
exp = shap.Explanation(sv[:,:,index], sv.base_values[:,index], test2[start_index:end_index], feature_names=test2.columns.tolist())
# generating plot
shap.waterfall_plot(exp[row],max_display=10,show= None)
plt.savefig('./plots/RF_Shap_Waterfall.png')
plt.clf()

shap.summary_plot(shap_values = shap_values[0],features = test[start_index:end_index],show=False,feature_names=test2.columns.tolist())
plt.savefig('./plots/ADA_Shap_Summary_Beeswarms.png')
