# Generated from: paddy.ipynb
# Converted at: 2026-03-30T16:23:41.942Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np

df=pd.read_csv('paddydataset.csv')
df.head(10)

df.info()

df.columns = df.columns.str.strip()

x=df.drop(columns='Paddy yield(in Kg)')
y=df['Paddy yield(in Kg)']
y

num_col=x.select_dtypes(exclude=('object')).columns
num_col

cat_col=x.select_dtypes(include=('object')).columns
cat_col

df.isnull().sum()

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


num_pipeline=Pipeline([
    ('scaler',StandardScaler())
])
num_pipeline

cat_pipeline=Pipeline([
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])
cat_pipeline

preprocessing=ColumnTransformer([
    ('num',num_pipeline,num_col),
    ('cat',cat_pipeline,cat_col)
])
preprocessing

from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestRegressor

pipeline=Pipeline([
    ('processing',preprocessing),
    ('model',RandomForestRegressor())
])
pipeline

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

pipeline.fit(x_train,y_train)

import joblib

joblib.dump(pipeline,'paddy.pkl')
print('pkl file saved successfully')