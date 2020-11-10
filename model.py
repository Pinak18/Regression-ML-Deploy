import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import pickle

dat = pd.read_csv('Linear_Reg_Dat.csv')

dat.dropna(subset = ['Salary'], axis=0, inplace=True)

z = np.abs(stats.zscore(dat))
dat = dat[(z < 3).all(axis=1)]

x = dat[['YearsExperience','InterviewScore']]
y = dat['Salary'] 

regressor = LinearRegression()
regressor.fit(x, y)

pickle.dump(regressor, open('model.pkl','wb')) # save model to disk in plckle format.

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9]]))