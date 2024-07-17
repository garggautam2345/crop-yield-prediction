from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
ohe=OneHotEncoder(drop='first')
st=StandardScaler()
preprocessor=ColumnTransformer(
        transformers=[
            ('ONE HOT ENCODING',ohe,[0,1]),
            ('STANDARD SCALLER',st,[2,3,4,5])
        ],
        remainder='passthrough'
    )

def pre(df):
    x=df.drop(columns={'hg/ha_yield'})
    y=df['hg/ha_yield']
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test

def updated(df,x_train,x_test): 
    return preprocessor.fit_transform(x_train),preprocessor.transform(x_test)

def model(x_train_updated,y_train):
    knn=KNeighborsRegressor(8)
    return knn.fit(x_train_updated,y_train)

def prediction(Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,a):
    features=np.array([[Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]])
    feature=preprocessor.transform(features)
    predicted_value=a.predict(feature).reshape(1,-1)
    return predicted_value[0]




