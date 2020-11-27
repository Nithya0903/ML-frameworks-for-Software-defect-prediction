# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('cm1.csv')
X = dataset.iloc[:,:-1].values
X=X[400:498,:]
y = dataset.iloc[:, 21].values
y=y[400:498]

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 21)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print("explained variance {}".format(explained_variance))

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1)
plt.show()
#selecting n=16 

pca = PCA(n_components = 16)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# Fitting Neural Network - MLP Classifier to the Training set
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(6,),activation='identity',solver='sgd',random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
print(y_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Finding Area Under ROC curve
from sklearn.metrics import roc_auc_score as roc
a = roc(y_test, y_pred, average='micro')

####################################################################

#Processing data for plotting graph

from sklearn.preprocessing import label_binarize
y = label_binarize(y, classes=[0, 1])
n_classes = y.shape[1]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],     remainder='passthrough')
y = np.array(columnTransformer.fit_transform(y),dtype=np.int)

y_test = label_binarize(y_test, classes=[0, 1])
y_test = np.array(columnTransformer.fit_transform(y_test),dtype=np.int)

y_pred = label_binarize(y_pred, classes=[0, 1])
y_pred = np.array(columnTransformer.fit_transform(y_pred),dtype=np.int)

# Compute ROC curve and ROC area for each class
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic cm1_PCA')
plt.legend(loc="lower right")
plt.show()