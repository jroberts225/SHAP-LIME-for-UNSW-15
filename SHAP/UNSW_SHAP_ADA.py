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


from sklearn.ensemble import AdaBoostClassifier
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
y = df.pop('label')
X = df
# summarize class distribution
counter = Counter(y)
print('after removing duplicates:',counter)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .70
print(df.head())

train, test = df[df['is_train']==True], df[df['is_train']==False]
print('Number of the training data:', len(train))
print('Number of the testing data:', len(test))

features = df.columns[:len(req_cols)-2]
print(features)

y_train, label = pd.factorize(train['Label'])
print(y_train)

y_test, label = pd.factorize(test['Label'])

X_train = np.array(train[features])
print(X_train)

abc = AdaBoostClassifier(n_estimators=50,learning_rate=0.5)
#----------------------------------------------------------------------------------------------------------
#Running the model

#START TIMER MODEL
start = time.time()
model = abc.fit(X_train, y_train)
#END TIMER MODEL
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')

#----------------------------------------------------------------------------------------------------------
#Data preprocessing
X_test = np.array(test[features])

#----------------------------------------------------------------------------------------------------------
# Model predictions 

#START TIMER PREDICTION
start = time.time()

y_pred = model.predict(X_test)

#END TIMER PREDICTION
end = time.time()
print('ELAPSE TIME PREDICTION: ',(end - start)/60, 'min')

#----------------------------------------------------------------------------------------------------------

y_test, label2 = pd.factorize(test['Label'])

pred_label = label[y_pred]

#----------------------------------------------------------------------------------------------------------
# Confusion Matrix
print('---------------------------------------------------------------------------------')
print('Generating Confusion Matrix')
print('---------------------------------------------------------------------------------')
print('')

confusion_matrix = pd.crosstab(test['Label'], pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(test['Label']))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
print(confusion_matrix)

#True positives and False positives and negatives
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
#Sum each Labels TP,TN,FP,FN in one overall measure
TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

#data preprocessin because numbers are getting big
TP_total = np.array(TP_total,dtype=np.float64)
TN_total = np.array(TN_total,dtype=np.float64)
FP_total = np.array(FP_total,dtype=np.float64)
FN_total = np.array(FN_total,dtype=np.float64)

#----------------------------------------------------------------------------------------------------------

#Metrics measure overall
Acc = ACC(TP_total,TN_total, FP_total, FN_total)
Precision = PRECISION(TP_total, FP_total)
Recall = RECALL(TP_total, FN_total)
F1 = F1(Recall,Precision)
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)
print('Accuracy: ', Acc)
print('Precision: ', Precision )
print('Recall: ', Recall )
print('F1: ', F1 )
print('BACC: ', BACC)
print('MCC: ', MCC)

#----------------------------------------
y_score = abc.predict_proba(test[features])
y_test_bin = label_binarize(y_test,classes = [0,1,2,3,4,5,6,7,8,9])
n_classes = y_test_bin.shape[1]
# print('rocauc is ',roc_auc_score(y_test_bin,y_score, multi_class='ovr'))
# ## Summary Bar Plot Global
start_index = 0
end_index = 100
test.pop('Label')
test.pop('is_train')
print(label2)
explainer = shap.KernelExplainer(abc.predict_proba, test[start_index:end_index])

shap_values = explainer.shap_values(test[start_index:end_index])

shap.summary_plot(shap_values = shap_values,
                  features = test[start_index:end_index],class_names=[label[0],label[1],label[2],label[3],label[4],label[5],label[6],label[7],label[8],label[9]],show=False)
plt.savefig('./plots/ADA_Shap_Summary_global.png')
plt.clf()


shap.summary_plot(shap_values = shap_values[0],
                 features = test[start_index:end_index],
                  show=False)
plt.savefig('./plots/ADA_Shap_Summary_Beeswarms.png')
  