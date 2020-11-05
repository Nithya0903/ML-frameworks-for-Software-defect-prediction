def preprocess(ds):
    import numpy as np
    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv("data\\"+str(ds))
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:, 21].values
    
    print("No of +ve exAamples are:",sum(y))
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

    print("No of +ve exAamples in training data are:",sum(y_train))
    print("No of +ve examples in testing data are:",sum(y_test))
    return (X_train,X_test,y_test,y_train)
   
   

