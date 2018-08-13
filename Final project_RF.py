# %%%%%%%%%%%%% Machine Learning I %%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Talal Alzahrani, MD, MPH ------>Email: tsa@gwu.edu
# %%%%%%%%%%%%% Date %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Final version August - 01 - 2018
# %%%%%%%%%%%%% Exploratory Data Analysis  %%%%%%%%%%
# Importing the required packages
import pandas as pd
pd.set_option('display.max_columns', 100)

# Importing the data
adult = pd.read_csv('adult.csv', sep=',', index_col=0)

# printing the dataset shape
print ('-'*40 + 'Start Console' + '-'*40 + '\n')

print("Dataset No. of Rows: ", adult.shape[0])
print("Dataset No. of Columns: ", adult.shape[1])

# printing the dataset observations
print("Dataset first few rows:\n ")
print(adult.head())

print ('-'*80 + '\n')

# printing the struture of the dataset
print("Dataset info:\n ")
print(adult.info())
print ('-'*80 + '\n')

# printing the summary statistics of the dataset
print(adult.describe(include='all'))
print ('-'*80 + '\n')


# %%%%%%%%%%%%% Random Forest  %%%%%%%%%%%%%%%%%%%%%%%%%%
# Importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

adult = pd.read_csv('adult.csv', sep=',', index_col=0)

#split the dataset
# separate the predictor and target variable
RF_X = adult.values[:, 1:]
RF_Y = adult.values[:, 0]

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(RF_X, RF_Y, test_size=0.3, random_state=99)

#perform training with random forest with all columns
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)

#plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, adult.iloc[:, 1:].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()

#select features to perform training with random forest with k columns
# select the training dataset on k-features
newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:73]]

# select the testing dataset on k-features
newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:73]]


#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(newX_train, y_train)

#make predictions

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)


# calculate metrics gini model

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# calculate metrics entropy model
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)


# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = adult['MIEV'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()


# confusion matrix for entropy model

conf_matrix = confusion_matrix(y_test, y_pred_k_features)
class_names = adult['MIEV'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()

