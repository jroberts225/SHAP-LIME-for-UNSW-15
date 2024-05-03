%matplotlib inline

print('--------------------------------------------------')
print('LGBM sensor and shap with Lime')
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

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
#from sklearn.metrics import auc_score
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification

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
df = df.sample(frac=1,replace=True,random_state=1)

le = LabelEncoder()
label = le.fit_transform(df['proto'])
df.drop("proto",axis=1,inplace=True)
df["proto"] = label

encoded,label_list = pd.factorize(df['attack_cat'], sort=False)
df["attack_cat"] = encoded
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
# Normalize your dataframe
df_max_scaled = df.copy()
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col] / t
df = df_max_scaled.assign(Label=y)
df = df.fillna(0)

# Immediately after, ensure 'Label' is removed if not needed for modeling
if 'Label' in df.columns:
    df = df.drop(columns=['Label'])


print('---------------------------------------------------------------------------------')
print('Counting labels')
print('---------------------------------------------------------------------------------')
print('')

y = df.pop('label')
X = df

# count class distribution
counter = Counter(y)
print(counter)

df = X.assign( ALERT = y)

#Remove duplicates
df = df.drop_duplicates()

y = df.pop('ALERT')
X = df

df = df.assign( ALERT = y)
# summarize class distribution
counter = Counter(y)
print('after removing duplicates:',counter)

# Train and Test split
# Before splitting your data into train and test sets, drop the 'Label' column if it exists
if 'Label' in df.columns:
    df = df.drop(columns=['Label'])

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, y, train_size=0.80)
df = X.assign( ALERT = y)

# extracting label name
notused, y_labels = pd.factorize(y)
# Transforming numpy format list
y_labels = list(y_labels)

print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')
counter = Counter(labels_train)
print(counter)

# call balance operation until all labels have the same size
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        train, labels_train = oversample(train, labels_train)

counter = Counter(labels_train)
print('train len',counter)

# # After OverSampling training dataset

train = train.assign( ALERT = labels_train)

#Drop ALert column from train
train.pop('ALERT')
labels_train_number, labels_train_label = pd.factorize(labels_train)
labels_test_number, labels_test_label = pd.factorize(labels_test)

# # Oversampling and balancing test data

counter = Counter(labels_test)
print(counter)
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        test, labels_test = oversample(test, labels_test)

counter = Counter(labels_test)
print('test len ', counter)

#joining features and label
test = test.assign(ALERT = labels_test)

#Randomize df order
test = test.sample(frac = 1)

#Drop label column
labels_test = test.pop('ALERT')

train = train.to_numpy()

if 'ALERT' in train.columns:
    train = train.drop(columns=['ALERT'])
if 'Label' in train.columns:
    train = train.drop(columns=['Label'])
print('Training the model')
print('------------------------------------------------------------------------------')
#START TIMER MODEL
start = time.time()

# evaluate the model
model = LGBMClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, train, labels_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# fit the model on the whole dataset
model = LGBMClassifier()

model.fit(train, labels_train)
#END TIMER MODEL
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')
print('------------------------------------------------------------------------------')
#------------------------------------------------------------------------------
# Get the predicted class for each observation

start = time.time()

y_pred = model.predict(test)
print('prediction',y_pred)

p, label2 = pd.factorize(y_pred)
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')

print('---------------------------------------------------------------------------------')
print('Generating Confusion Matrix')
print('---------------------------------------------------------------------------------')
print('')

print(y_pred)
print(labels_test)
pred_label = y_pred

confusion_matrix = pd.crosstab(labels_test, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(labels_test))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
print(confusion_matrix)

#---------------------------------------------------------------------
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

#---------------------------------------------------------------------
TP_total = np.array(TP_total,dtype=np.float64)
TN_total = np.array(TN_total,dtype=np.float64)
FP_total = np.array(FP_total,dtype=np.float64)
FN_total = np.array(FN_total,dtype=np.float64)
Acc = ACC(TP_total,TN_total, FP_total, FN_total)
Precision = PRECISION(TP_total, FP_total)
Recall = RECALL(TP_total, FN_total)
F1 = F1(Recall,Precision)
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)
print('---------------------------------------------------------------------------------')

print('Accuracy total: ', Acc)
print('Precision total: ', Precision )
print('Recall total: ', Recall )
print('F1 total: ', F1 )
print('BACC total: ', BACC)
print('MCC total: ', MCC)
# print('rocauc is ',roc_auc_score(labels_test, model.predict_proba(test), multi_class='ovr'))

y_pred =model.predict_proba(test)
label = label2
#---------------------------------------------------------------------
classes_n = []
y_test = label
for i in label: classes_n.append(i)
y_test_bin = label_binarize(labels_test,classes = classes_n)
n_classes = y_test_bin.shape[1]


for i in range(0,len(label)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
print('---------------------------------------------------------------------------------')
# ## Summary Bar Plot Global
explainer = shap.TreeExplainer(model)
start_index = 0
end_index = 1000
shap_values = explainer.shap_values(test[start_index:end_index])

shap_obj = explainer(test[start_index:end_index])
shap.summary_plot(shap_values = shap_values,
                  features = test[start_index:end_index],
                 class_names=[y_labels[0],y_labels[1],'''y_labels[2],y_labels[3],y_labels[4],y_labels[5],y_labels[6],y_labels[7],y_labels[8],y_labels,[9]'''],show=True)
# plt.savefig('./Light_Shap_Summary_global2.png')
plt.clf()

shap_obj = explainer(test[start_index:end_index])
shap.summary_plot(shap_values = np.take(shap_obj.values,0,axis=-1),
                  features = test[start_index:end_index],show=True)
# plt.savefig('./Light_Shap_Summary_Beeswarms2.png')
plt.clf()