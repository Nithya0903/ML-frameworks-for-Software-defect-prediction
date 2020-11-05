# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('jm1.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 21].values

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,)
             ]
        }
       ]
clf = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='accuracy')
clf.fit(X,y)

print("Best parameters set found on development set:")
print(clf.best_params_)