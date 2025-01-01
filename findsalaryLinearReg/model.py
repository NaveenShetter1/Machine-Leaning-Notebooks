# importing the libraries

import numpy as np
import pandas as pd
import pickle

dataset=pd.read_excel(r'C:\Users\NaveenShetter\Desktop\hiring.xlsx')


# print(dataset.describe())
# print(dataset.head())

dataset['experience'].fillna(0,inplace=True)

# print(dataset)

dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

# print(dataset)
# we will take all the values as inputs or independent

x=dataset.iloc[:,:3]
y=dataset.iloc[:,-1]



# converting str to int

def con_to_int(word):
    word_dict={'one':1,'two':2,'three':3,'four':4,
    'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,0:0,'eleven':11,'twelve':12,'zero':0}

    return word_dict[word]


x['experience']=x['experience'].apply(lambda y: con_to_int(y))


# since we have less data we will use all as training data 

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

# fitting model with data
regressor.fit(x,y)

pickle.dump(regressor,open('model.pkl','wb'))
# loading model to compare the result

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[2,9,6]]))