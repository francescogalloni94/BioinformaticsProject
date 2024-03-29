import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from plots import plot_confusion_matrix
from plots import plotRoc_curve
from plots import plotPrecisionRecall_curve
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve




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

#Function that create the neural network. it is used in the sklearn wrapper of keras
def create_model(input_units,hidden_layers,hidden_units):
    model = Sequential()
    #Adding input and first hidden layer
    model.add(Dense(units=hidden_units,kernel_initializer='uniform',activation='relu',input_dim=input_units))
    #adding hidden layers
    for i in range(hidden_layers):
        model.add(Dense(units=hidden_units,kernel_initializer='uniform',activation='relu'))
    #Adding output layer
    model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    return model


#Training and Testing Active Inactive Enhancer Random Forest
print('Training Active Inactive Enhancer Random forest classifier\nwith grid search model selection and cross validation...\n')

A_I_Enhancer_parameters = [{'n_estimators':[10,50,100,200],'criterion':['entropy']}]
A_I_Enhancer_classifier = GridSearchCV(estimator=RandomForestClassifier(),param_grid=A_I_Enhancer_parameters,scoring='f1',cv=3,n_jobs=-1)
A_I_Enhancer_classifier = A_I_Enhancer_classifier.fit(A_I_Enhancer_X_train,A_I_Enhancer_y_train)
print('Best parameters value: '+str(A_I_Enhancer_classifier.best_params_)+'\n')
print('Best scores on 3-Fold cross validation: '+str(A_I_Enhancer_classifier.best_score_)+'\n')
y_AI_Enhancer_pred = A_I_Enhancer_classifier.predict(A_I_Enhancer_X_test)
AI_Enhancer_probs = A_I_Enhancer_classifier.predict_proba(A_I_Enhancer_X_test)
AI_Enhancer_probs = AI_Enhancer_probs[:,1]
y_AI_Enhancer_pred_labels = A_I_Enhancer_encoder.inverse_transform(y_AI_Enhancer_pred)
y_AI_Enhancer_test_labels = A_I_Enhancer_encoder.inverse_transform(A_I_Enhancer_y_test)

cm = confusion_matrix(y_AI_Enhancer_test_labels,y_AI_Enhancer_pred_labels)
print('Confusion Matrix:\n')
print(cm)
print('Accuracy score: '+str(accuracy_score(A_I_Enhancer_y_test,y_AI_Enhancer_pred)))
print('F1 score: '+str(f1_score(A_I_Enhancer_y_test,y_AI_Enhancer_pred))+'\n')
plot_confusion_matrix(cm,filename='A_I_Enhancer_RF_cm.png',target_names=['A-E','I-E'],title='Active Inactive Enhancer Random Forest')
fpr, tpr, roc_threshold = roc_curve(A_I_Enhancer_y_test,AI_Enhancer_probs)
precision, recall, precision_thresholds = precision_recall_curve(A_I_Enhancer_y_test,AI_Enhancer_probs)
roc_auc = auc(fpr,tpr)
pr_auc = auc(recall, precision)
print('AUROC: '+str(roc_auc))
print('AUPRC: '+str(pr_auc))
plotRoc_curve(fpr,tpr,roc_auc,'AI_Enhancer_RF_roc.png','ROC curve Active Inactive Enhancer Random Forest')
plotPrecisionRecall_curve(precision,recall,pr_auc,'AI_Enhancer_RF_pr.png','P-R curve Active Inactive Enhancer Random Forest')


#Training and testing Active Inactive Enhancer Neural Network
print('Training Active Inactive Enhancer Neural Network')
keras_classifier = KerasClassifier(build_fn=create_model,input_units=0,hidden_layers=0,hidden_units=0)
param_grid = [{'input_units':[A_I_Enhancer_X_train.shape[1]],'hidden_layers':[1,2,3],'hidden_units':[10,20,50],
               'batch_size':[1000],'epochs':[100]}]
AIE_neural_network = GridSearchCV(estimator=keras_classifier,param_grid=param_grid,scoring='f1',n_jobs=-1,cv=3)
AIE_neural_network = AIE_neural_network.fit(A_I_Enhancer_X_train,A_I_Enhancer_y_train)
print('best neural network parameters are: \n')
print(AIE_neural_network.best_params_)
print('best accuracy on 3-fold cross validation: '+str(AIE_neural_network.best_score_))

y_AI_Enhancer_pred = AIE_neural_network.predict(A_I_Enhancer_X_test)
AI_Enhancer_probs = AIE_neural_network.predict_proba(A_I_Enhancer_X_test)
AI_Enhancer_probs = AI_Enhancer_probs[:,1]
y_AI_Enhancer_pred_labels = A_I_Enhancer_encoder.inverse_transform(y_AI_Enhancer_pred)
y_AI_Enhancer_test_labels = A_I_Enhancer_encoder.inverse_transform(A_I_Enhancer_y_test)

cm = confusion_matrix(y_AI_Enhancer_test_labels,y_AI_Enhancer_pred_labels)
print('Confusion Matrix:\n')
print(cm)
plot_confusion_matrix(cm,filename='A_I_Enhancer_NN_cm.png',target_names=['A-E','I-E'],title='Active Inactive Enhancer Neural Network')
print('Accuracy score: '+str(accuracy_score(A_I_Enhancer_y_test,y_AI_Enhancer_pred)))
print('F1 score: '+str(f1_score(A_I_Enhancer_y_test,y_AI_Enhancer_pred))+'\n')
fpr, tpr, roc_threshold = roc_curve(A_I_Enhancer_y_test,AI_Enhancer_probs)
precision, recall, precision_thresholds = precision_recall_curve(A_I_Enhancer_y_test,AI_Enhancer_probs)
roc_auc = auc(fpr,tpr)
pr_auc = auc(recall, precision)
print('AUROC: '+str(roc_auc))
print('AUPRC: '+str(pr_auc))
plotRoc_curve(fpr,tpr,roc_auc,'AI_Enhancer_NN_roc.png','ROC curve Active Inactive Enhancer Neural Network')
plotPrecisionRecall_curve(precision,recall,pr_auc,'AI_Enhancer_NN_pr.png','P-R curve Active Inactive Enhancer Neural Network')


#Training and Testing Active Inactive Promoter Random Forest
print('Training Active Inactive Promoter Random forest classifier\n with grid search model selection and cross validation...\n')
A_I_Promoter_parameters = [{'n_estimators':[10,50,100,200],'criterion':['entropy']}]
A_I_Promoter_classifier = GridSearchCV(estimator=RandomForestClassifier(),param_grid=A_I_Promoter_parameters,scoring='f1',cv=3)
A_I_Promoter_classifier = A_I_Promoter_classifier.fit(A_I_Promoter_X_train,A_I_Promoter_y_train)
print('Best parameters value: '+str(A_I_Promoter_classifier.best_params_)+'\n')
print('Best scores on 3-Fold cross validation: '+str(A_I_Promoter_classifier.best_score_)+'\n')

y_AI_Promoter_pred = A_I_Promoter_classifier.predict(A_I_Promoter_X_test)
AI_Promoter_probs = A_I_Promoter_classifier.predict_proba(A_I_Promoter_X_test)
AI_Promoter_probs = AI_Promoter_probs[:,1]
y_AI_Promoter_pred_labels = A_I_Promoter_encoder.inverse_transform(y_AI_Promoter_pred)
y_AI_Promoter_test_labels = A_I_Promoter_encoder.inverse_transform(A_I_Promoter_y_test)

cm = confusion_matrix(y_AI_Promoter_test_labels,y_AI_Promoter_pred_labels)
print('Confusion Matrix:\n')
print(cm)
plot_confusion_matrix(cm,filename='A_I_Promoter_RF_cm.png',target_names=['A-P','I-P'],title='Active Inactive Promoter Random Forest')
print('Accuracy score: '+str(accuracy_score(A_I_Promoter_y_test,y_AI_Promoter_pred)))
print('F1 score: '+str(f1_score(A_I_Promoter_y_test,y_AI_Promoter_pred))+'\n')
fpr, tpr, roc_threshold = roc_curve(A_I_Promoter_y_test,AI_Promoter_probs)
precision, recall, precision_thresholds = precision_recall_curve(A_I_Promoter_y_test,AI_Promoter_probs)
roc_auc = auc(fpr,tpr)
pr_auc = auc(recall, precision)
print('AUROC: '+str(roc_auc))
print('AUPRC: '+str(pr_auc))
plotRoc_curve(fpr,tpr,roc_auc,'AI_Promoter_RF_roc.png','ROC curve Active Inactive Promoter Random Forest')
plotPrecisionRecall_curve(precision,recall,pr_auc,'AI_Promoter_RF_pr.png','P-R curve Active Inactive Promoter Random Forest')



#Training and testing Active Inactive Promoter Neural Network
print('Training Active Inactive Promoter Neural Network')
keras_classifier = KerasClassifier(build_fn=create_model,input_units=0,hidden_layers=0,hidden_units=0)
param_grid = [{'input_units':[A_I_Promoter_X_train.shape[1]],'hidden_layers':[1,2,3],'hidden_units':[10,20,50],
               'batch_size':[1000],'epochs':[100]}]
AIP_neural_network = GridSearchCV(estimator=keras_classifier,param_grid=param_grid,scoring='f1',n_jobs=-1,cv=3)
AIP_neural_network = AIP_neural_network.fit(A_I_Promoter_X_train,A_I_Promoter_y_train)
print('best neural network parameters are: \n')
print(AIP_neural_network.best_params_)
print('best accuracy on 3-fold cross validation: '+str(AIP_neural_network.best_score_))

y_AI_Promoter_pred = AIP_neural_network.predict(A_I_Promoter_X_test)
AI_Promoter_probs = AIP_neural_network.predict_proba(A_I_Promoter_X_test)
AI_Promoter_probs = AI_Promoter_probs[:,1]
y_AI_Promoter_pred_labels = A_I_Promoter_encoder.inverse_transform(y_AI_Promoter_pred)
y_AI_Promoter_test_labels = A_I_Promoter_encoder.inverse_transform(A_I_Promoter_y_test)

cm = confusion_matrix(y_AI_Promoter_test_labels,y_AI_Promoter_pred_labels)
print('Confusion Matrix:\n')
print(cm)
plot_confusion_matrix(cm,filename='A_I_Promoter_NN_cm.png',target_names=['A-P','I-P'],title='Active Inactive Promoter Neural Network')
print('Accuracy score: '+str(accuracy_score(A_I_Promoter_y_test,y_AI_Promoter_pred)))
print('F1 score: '+str(f1_score(A_I_Promoter_y_test,y_AI_Promoter_pred))+'\n')
fpr, tpr, roc_threshold = roc_curve(A_I_Promoter_y_test,AI_Promoter_probs)
precision, recall, precision_thresholds = precision_recall_curve(A_I_Promoter_y_test,AI_Promoter_probs)
roc_auc = auc(fpr,tpr)
pr_auc = auc(recall, precision)
print('AUROC: '+str(roc_auc))
print('AUPRC: '+str(pr_auc))
plotRoc_curve(fpr,tpr,roc_auc,'AI_Promoter_NN_roc.png','ROC curve Active Inactive Promoter Neural Network')
plotPrecisionRecall_curve(precision,recall,pr_auc,'AI_Promoter_NN_pr.png','P-R curve Active Inactive Promoter Neural Network')


#Training and testing Active Enhancer Active Promoter Random Forest
print('Training Active Enhancer, Active Promoter Random forest classifier\n with grid search model selection and cross validation...\n')
A_Enh_Prom_parameters = [{'n_estimators':[10,50,100,200],'criterion':['entropy']}]
A_Enh_Prom_classifier = GridSearchCV(estimator=RandomForestClassifier(),param_grid=A_Enh_Prom_parameters,scoring='f1',cv=3)
A_Enh_Prom_classifier = A_Enh_Prom_classifier.fit(A_Enh_Prom_X_train,A_Enh_Prom_y_train)
print('Best parameters value: '+str(A_Enh_Prom_classifier.best_params_)+'\n')
print('Best scores on 3-Fold cross validation: '+str(A_Enh_Prom_classifier.best_score_)+'\n')

y_A_Enh_Prom_pred = A_Enh_Prom_classifier.predict(A_Enh_Prom_X_test)
A_Enh_Prom_probs = A_Enh_Prom_classifier.predict_proba(A_Enh_Prom_X_test)
A_Enh_Prom_probs = A_Enh_Prom_probs[:,1]
y_A_Enh_Prom_pred_labels = A_Enh_Prom_encoder.inverse_transform(y_A_Enh_Prom_pred)
y_A_Enh_Prom_test_labels = A_Enh_Prom_encoder.inverse_transform(A_Enh_Prom_y_test)

cm = confusion_matrix(y_A_Enh_Prom_test_labels,y_A_Enh_Prom_pred_labels)
print('Confusion Matrix:\n')
print(cm)
plot_confusion_matrix(cm,filename='A_Enh_Prom_RF_cm.png',target_names=['A-E','A-P'],title='Active Enhancer Promoter Random Forest')
print('Accuracy score: '+str(accuracy_score(A_Enh_Prom_y_test,y_A_Enh_Prom_pred)))
print('F1 score: '+str(f1_score(A_Enh_Prom_y_test,y_A_Enh_Prom_pred))+'\n')
fpr, tpr, roc_threshold = roc_curve(A_Enh_Prom_y_test,A_Enh_Prom_probs)
precision, recall, precision_thresholds = precision_recall_curve(A_Enh_Prom_y_test,A_Enh_Prom_probs)
roc_auc = auc(fpr,tpr)
pr_auc = auc(recall, precision)
print('AUROC: '+str(roc_auc))
print('AUPRC: '+str(pr_auc))
plotRoc_curve(fpr,tpr,roc_auc,'A_Enh_Prom_RF_roc.png','ROC curve Active Enhancer Active Promoter Random Forest')
plotPrecisionRecall_curve(precision,recall,pr_auc,'A_Enh_Prom_RF_pr.png','P-R curve Active Enhancer Active Promoter Random Forest')

#function to training and testing Active Enhancer  Active Promoter Neural Network.
def A_Enh_prom_NeuralNetwork(X_train,y_train,balanced=False):

    if balanced:
        name = 'balanced'
    else:
        name = ''

    print('Training Active Enhancer Active Promoter Neural Network '+name)
    keras_classifier = KerasClassifier(build_fn=create_model,input_units=0,hidden_layers=0,hidden_units=0)
    param_grid = [{'input_units':[X_train.shape[1]],'hidden_layers':[1,2,3],'hidden_units':[10,20,50],
                   'batch_size':[1000],'epochs':[100]}]
    AEP_neural_network = GridSearchCV(estimator=keras_classifier,param_grid=param_grid,scoring='f1',n_jobs=-1,cv=3)
    AEP_neural_network = AEP_neural_network.fit(X_train,y_train)
    print('best neural network parameters are: \n')
    print(AEP_neural_network.best_params_)
    print('best accuracy on 3-fold cross validation: '+str(AEP_neural_network.best_score_))

    y_A_Enh_Prom_pred = AEP_neural_network.predict(A_Enh_Prom_X_test)
    A_Enh_Prom_probs = AEP_neural_network.predict_proba(A_Enh_Prom_X_test)
    A_Enh_Prom_probs = A_Enh_Prom_probs[:,1]
    y_A_Enh_Prom_pred_labels = A_Enh_Prom_encoder.inverse_transform(y_A_Enh_Prom_pred)
    y_A_Enh_Prom_test_labels = A_Enh_Prom_encoder.inverse_transform(A_Enh_Prom_y_test)

    cm = confusion_matrix(y_A_Enh_Prom_test_labels,y_A_Enh_Prom_pred_labels)
    print('Confusion Matrix:\n')
    print(cm)
    plot_confusion_matrix(cm,filename='A_Enh_Prom_NN_cm_'+name+'.png',target_names=['A-P','A-E'],title='Active Enhancer Active Promoter Neural Network '+name)
    print('Accuracy score: '+str(accuracy_score(A_Enh_Prom_y_test,y_A_Enh_Prom_pred)))
    print('F1 score: '+str(f1_score(A_Enh_Prom_y_test,y_A_Enh_Prom_pred))+'\n')
    fpr, tpr, roc_threshold = roc_curve(A_Enh_Prom_y_test,A_Enh_Prom_probs)
    precision, recall, precision_thresholds = precision_recall_curve(A_Enh_Prom_y_test,A_Enh_Prom_probs)
    roc_auc = auc(fpr,tpr)
    pr_auc = auc(recall, precision)
    print('AUROC: '+str(roc_auc))
    print('AUPRC: '+str(pr_auc))
    plotRoc_curve(fpr,tpr,roc_auc,'A_Enh_Prom_NN_roc_'+name+'.png','ROC curve Active Enhancer Active Promoter Neural Network '+name)
    plotPrecisionRecall_curve(precision,recall,pr_auc,'A_Enh_Prom_NN_pr_'+name+'.png','P-R curve Active Enhancer Active Promoter Neural Network '+name)


#training neural network on imbalanced data
A_Enh_prom_NeuralNetwork(A_Enh_Prom_X_train,A_Enh_Prom_y_train)

#Trying balancing data
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_resampled, Y_resampled = ros.fit_resample(A_Enh_Prom_X_train,A_Enh_Prom_y_train)

#training neural network on balanced data
A_Enh_prom_NeuralNetwork(X_resampled,Y_resampled,balanced=True)