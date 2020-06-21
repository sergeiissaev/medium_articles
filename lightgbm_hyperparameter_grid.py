import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import pprint
RANDOM_STATE = 42# Importing the dataset
dataset = pd.read_csv('dataframe.csv')
#One hot encoding
df = pd.concat([dataset,pd.get_dummies(dataset['categorical_var'], prefix='categorical_var')],axis=1)#Select columns
X = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 24, 26, 27, 30, 31, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]].values
y = df.iloc[:, list(dataset.columns).index('seconds')].values# Splitting the dataset into the training set and test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = RANDOM_STATE)# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Set the minimum error arbitrarily large
min = 99999999999999999999999 
count = 0 #Used for keeping track of the iteration number
#How many runs to perform using randomly selected hyperparameters
iterations = 50
for i in range(iterations):
    print('iteration number', count)
    count += 1 #increment count
    try:
        d_train = lgb.Dataset(x_train, label=y_train) #Load in data
        params = {} #initialize parameters
        params['learning_rate'] = np.random.uniform(0, 1)
        params['boosting_type'] = np.random.choice(['gbdt', 'dart', 'goss'])
        params['objective'] = 'regression'
        params['metric'] = 'mae'
        params['sub_feature'] = np.random.uniform(0, 1)
        params['num_leaves'] = np.random.randint(20, 300)
        params['min_data'] = np.random.randint(10, 100)
        params['max_depth'] = np.random.randint(5, 200)
        iterations = np.random.randint(10, 10000)
        print(params, iterations)#Train using selected parameters
clf = lgb.train(params, d_train, iterations)y_pred=clf.predict(x_test) #Create predictions on test setmae=mean_absolute_error(y_pred,y_test)
        print('MAE:', mae)
        if mae < min:
            min = mae
            pp = params 
    except: #in case something goes wrong
        print('failed with')
        print(params)
print("*" * 50)
print('Minimum is: ', min)
print('Used params', pp)
