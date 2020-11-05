# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
def my_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)
# Importing the dataset
dataset = pd.read_csv('cm1.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
#Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0,sampling_strategy=0.5)
X_train,y_train = rus.fit_resample(X_train, y_train)

'''
#Oversampling
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(random_state=0,sampling_strategy=0.5)
X_train, y_train = oversample.fit_resample(X_train, y_train)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying Dimensionality reduction
param_grid = [{
        "gamma": np.linspace(0.03, 0.05, 10),
        "kernel": ["rbf", "sigmoid", "linear", "poly"]
    }]

kpca=KernelPCA(fit_inverse_transform=True, n_jobs=-1) 
grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=my_scorer)
grid_search.fit(X)

print("Optimal parameters are:")
print(grid_search.best_params_)