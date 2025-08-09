import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler ,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression

# 1.Load the data
housing = pd.read_csv("housing.csv")

# 2.Create a StratifiedShuffleSplit
housing['income_cat'] = pd.cut(housing['median_income'],bins =[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2)
for train_index , test_index in split.split(housing,housing['income_cat']):
    strat_train  = housing.loc[train_index].drop('income_cat',axis=1)
    strat_test = housing.loc[test_index].drop('income_cat',axis=1)

# we will work on the copy of training set
housing = strat_train.copy()

# 3.separate features and labels
housing_label = housing['median_house_value'].copy()
housing = housing.drop('median_house_value',axis=1)

# print(housing,housing_label)

# 4.Separate numerical & Categorical columns
num_atrribs = housing.drop('ocean_proximity',axis=1).columns.tolist()
cat_attribs = ['ocean_proximity']

# 5. lets build the pipeline

#numerical Data
num_pipeline = Pipeline([
    ("Imputer", SimpleImputer(strategy='median')),
    ("Scaler" , StandardScaler())
])

# Cat Data
cat_pipeline = Pipeline([
    ("HotEncoder",OneHotEncoder(handle_unknown='ignore'))
])

#Construct the full pipeline
fullpipeline =ColumnTransformer([
    ("nums", num_pipeline,num_atrribs),
    ("cats",cat_pipeline,cat_attribs)
])

# transfrom the data 
housing_prepared = fullpipeline.fit_transform(housing)
print(housing_prepared)

# Train the model

#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_label)
lin_pred = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_label,lin_pred)
lin_rmse = -cross_val_score(lin_reg,housing_prepared,housing_label,scoring="neg_root_mean_squared_error",cv=10)
# print(f"The root_mean_squared_error of Linear Regression is {lin_rmse}")
print("The cross_val_score of Linear Regression --"
      ,pd.Series(lin_rmse).describe())

#Decision Regressor
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_label)
dec_pred = dec_reg.predict(housing_prepared)
# dec_rmse = root_mean_squared_error(dec_pred,housing_label)
# print(f"The root_mean_squared_error of Decision Regressor is {dec_rmse}")
dec_rmse = -cross_val_score(dec_reg,housing_prepared,housing_label,scoring="neg_root_mean_squared_error",cv=10)
print("The cross_val_score of Decision Tree is --",pd.Series(dec_rmse).describe())

#Random Forest Regressor
ran_reg= RandomForestRegressor()
ran_reg.fit(housing_prepared,housing_label)
ran_pred = ran_reg.predict(housing_prepared)
# ran_rmse = root_mean_squared_error(ran_pred,housing_label)
# print(f"The root_mean_squared_error of Random Forest Regressor is {ran_rmse}")
ran_rmse = -cross_val_score(ran_reg,housing_prepared,housing_label,scoring="neg_root_mean_squared_error",cv=10)
print("The cross_val_score of Random Forest is --",pd.Series(ran_rmse).describe())