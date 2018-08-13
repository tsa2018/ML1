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

# %%%%%%%%%%%%% Support vector machine  %%%%%%%%%%%%%%%%%%%%%%%%%%
# Importing the required packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

adult = pd.read_csv('adult.csv', sep=',', index_col=0)

# encoding the features using get dummies
X_data = pd.get_dummies(adult.iloc[:, 1:])
X = X_data.values

# encoding the class with sklearn's LabelEncoder
Y_data = adult.values[:, 0]

class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(Y_data)

# Spliting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# creating the classifier object
clf = SVC()

# performing training
clf.fit(X_train, y_train)

# predicton on test
y_pred = clf.predict(X_test)

print("\n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = adult['MIEV'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()


