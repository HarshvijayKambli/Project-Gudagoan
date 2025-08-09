import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

model_file = "model.pkl"
pipeline_file = "pipeline.pkl"

housing = pd.read_csv('housing.csv')

def build_pipeline(num_attribs,cat_attribs):
    # for numerical 
    num_pipeline  =Pipeline([
        ("Impute" , SimpleImputer( strategy="mean")),
        ("Scaler",StandardScaler())
    ])

    # for categorical
    cat_pipeline =Pipeline ([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    # full pipeline
    fullpipeline = ColumnTransformer([
        ("nums",num_pipeline,num_attribs),
        ("Cat", cat_pipeline,cat_attribs)
    ])

    return fullpipeline

if not os.path.exists(model_file):
    housing['income_cat'] = pd.cut(housing['median_income'],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index ,test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop('income_cat',axis=1).to_csv('Input.csv',index=False) 
        housing= housing.loc[train_index].drop('income_cat',axis=1)
         
    housing_label = housing['median_house_value'].copy()
    housing_features = housing.drop('median_house_value',axis=1)

    num_atrribs = housing_features.drop('ocean_proximity',axis=1).columns.tolist()
    cat_attribs = ['ocean_proximity']

    # print(housing_features.columns.tolist())
    # print(cat_attribs)

    pipeline = build_pipeline(
        num_atrribs,cat_attribs
    )
    housing_prepared = pipeline.fit_transform(housing_features)
     
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_label)

    joblib.dump(model,model_file)   
    joblib.dump(pipeline ,pipeline_file)
    print("Model is train sucessfully")

else:
    model = joblib.load(model_file)
    pipeline = joblib.load(pipeline_file)
    
    Input_data = pd.read_csv('Input.csv')
    transformed_input = pipeline.transform(Input_data)
    predicition = model.predict(transformed_input)
    Input_data['median_house_value'] = predicition

    Input_data.to_csv("output.csv",index=False)
    print("Inference is complete , results saved to output.csv")    