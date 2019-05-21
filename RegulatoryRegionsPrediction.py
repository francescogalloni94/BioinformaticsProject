import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Read data and data preprocessing
print('Reading data and preprocessing..\n')
epigenomic_data = pd.read_csv('GM12878.csv')
classes = pd.read_csv('GM12878Classes.csv',header=None)
epigenomic_data = epigenomic_data.iloc[:,1:]
frames = [epigenomic_data,classes]
concat_data = pd.concat(frames,axis=1)
concat_data = concat_data.rename(columns={ concat_data.columns[-1]: "class" })

#Complete data with all classes
complete_data_X = concat_data.iloc[:,:-1].values
complete_data_y = concat_data.iloc[:,-1].values

#Active Inactive Enhancer data
A_I_Enhancer_data = concat_data.loc[(concat_data['class'] == 'A-E') | (concat_data['class'] == 'I-E')]
A_I_Enhancer_X = A_I_Enhancer_data.iloc[:,:-1].values
A_I_Enhancer_y = A_I_Enhancer_data.iloc[:,-1].values

#Active Inactive Promoter data
A_I_Promoter_data = concat_data.loc[(concat_data['class']  == 'A-P') | (concat_data['class'] == 'I-P')]
A_I_Promoter_X = A_I_Promoter_data.iloc[:,:-1].values
A_I_Promoter_y = A_I_Promoter_data.iloc[:,-1].values

#Active Enhancer Active Promoter data
A_Enh_Prom_data = concat_data.loc[(concat_data['class'] == 'A-E') | (concat_data['class'] == 'A-P')]
A_Enh_Prom_X = A_Enh_Prom_data.iloc[:,:-1].values
A_Enh_Prom_y = A_Enh_Prom_data.iloc[:,-1].values

#Encoding Y variables
complete_data_encoder = LabelEncoder()
complete_data_y = complete_data_encoder.fit_transform(complete_data_y)
A_I_Enhancer_encoder = LabelEncoder()
A_I_Enhancer_y = A_I_Enhancer_encoder.fit_transform(A_I_Enhancer_y)
A_I_Promoter_encoder = LabelEncoder()
A_I_Promoter_y = A_I_Promoter_encoder.fit_transform(A_I_Promoter_y)
A_Enh_Prom_encoder = LabelEncoder()
A_Enh_Prom_y = A_Enh_Prom_encoder.fit_transform(A_Enh_Prom_y)

#Splittin in training and test set every task
print('Splitting in Training and Test set...\n')
complete_data_X_train,complete_data_X_test,complete_data_y_train,complete_data_y_test = train_test_split(complete_data_X,complete_data_y,test_size=0.3,shuffle=True)
A_I_Enhancer_X_train,A_I_Enhancer_X_test,A_I_Enhancer_y_train,A_I_Enhancer_y_test = train_test_split(A_I_Enhancer_X,A_I_Enhancer_y,test_size=0.3,shuffle=True)
A_I_Promoter_X_train,A_I_Promoter_X_test,A_I_Promoter_y_train,A_I_Promoter_y_test = train_test_split(A_I_Promoter_X,A_I_Promoter_y,test_size=0.3,shuffle=True)
A_Enh_Prom_X_train,A_Enh_Prom_X_test,A_Enh_Prom_y_train,A_Enh_Prom_y_test = train_test_split(A_Enh_Prom_X,A_Enh_Prom_y,test_size=0.3,shuffle=True)

#Performing feature scaling
print('Performing Feature Scaling...\n')
sc_complete_data = StandardScaler()
complete_data_X_train = sc_complete_data.fit_transform(complete_data_X_train)
complete_data_X_test = sc_complete_data.transform(complete_data_X_test)
sc_AIE = StandardScaler()
A_I_Enhancer_X_train = sc_AIE.fit_transform(A_I_Enhancer_X_train)
A_I_Enhancer_X_test = sc_AIE.transform(A_I_Enhancer_X_test)
sc_AIP = StandardScaler()
A_I_Promoter_X_train = sc_AIP.fit_transform(A_I_Promoter_X_train)
A_I_Promoter_X_test = sc_AIP.transform(A_I_Promoter_X_test)
sc_AEP = StandardScaler()
A_Enh_Prom_X_train = sc_AEP.fit_transform(A_Enh_Prom_X_train)
A_Enh_Prom_X_test = sc_AEP.transform(A_Enh_Prom_X_test)

