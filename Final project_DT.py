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



# %%%%%%%%%%%%% Decision Tree %%%%%%%%%%%%%%%%%%%%%%%%
#  Importing the required packages
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser
import warnings
warnings.filterwarnings("ignore")

# Split the dataset
DT_X = adult.values[:, 1:]
DT_Y = adult.values[:, 0]
print(DT_X)
print(DT_Y)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(DT_X, DT_Y, test_size=0.3, random_state=99)

# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)

# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)

# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)

# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')



# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')



# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = adult.MIEV.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = adult.MIEV.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


# End DT code
