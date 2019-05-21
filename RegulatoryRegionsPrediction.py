import pandas as pd
import numpy as np
import matplotlib as plt

#Read data and data preprocessing
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



