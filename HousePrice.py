"""
HOUSE PRICE PREDICTION

https://www.kaggle.com/anmolkumar/house-price-prediction-challenge

POSTED_BY          - Category marking who has listed the property
UNDER_CONSTRUCTION - Under Construction or Not
RERA               - Rera approved or Not
BHK_NO             - Number of Rooms
BHKORRK            - Type of property
SQUARE_FT          - Total area of the house in square feet
READYTOMOVE        - Category marking Ready to move or Not
RESALE             - Category marking Resale or not
ADDRESS            - Address of the property
LONGITUDE          - Longitude of the property
LATITUDE           - Latitude of the property

"""

#IMPORTS

#preprocessing
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#geo
from geopy.geocoders import Nominatim
import folium
from folium import plugins

#models
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost
import lightgbm as lgbm

#metrics
from sklearn.metrics import mean_absolute_error

#Plots
import matplotlib.pyplot as plt

#feature engineering
from sklearn.feature_selection import RFE

#model selection
from sklearn.model_selection import GridSearchCV

#Select Directory
os.chdir('C:\DataSets\HousePrice')

#Load Data
ds = pd.read_csv('train.csv')

#Pre-Processing
ds.insert(9, 'CITY', ds['ADDRESS'].str.extract(',(\w.*)'))
ds.insert(9, 'NEIGHBORHOOD', ds['ADDRESS'].str.extract('(\w.*),'))

POSTED = pd.get_dummies(ds['POSTED_BY'])
ds = pd.merge(ds.reset_index(), POSTED.reset_index())

label = LabelEncoder()
ds['CITY'] = label.fit_transform(ds['CITY'])
ds['NEIGHBORHOOD'] = label.fit_transform(ds['NEIGHBORHOOD'])

ds = ds[ds['TARGET(PRICE_IN_LACS)'] < 1000]

ds.insert(11, 'geoloc', ds['LONGITUDE'].astype(str) + ', ' + ds['LATITUDE'].astype(str))

geolocator = Nominatim(user_agent="HousePrices")
ds.insert(12, 'Nominatim', ds['geoloc'].apply(lambda x: (geolocator.reverse(x, timeout=4))))

def notnull(x, y):
    try:
        return x.raw['address'][y]
    except:
        return ''
 
ds.insert(13, 'Country', ds['Nominatim'].apply(lambda x: notnull(x,'country')))
ds.insert(14, 'State', ds['Nominatim'].apply(lambda x: notnull(x,'state')))

ds = ds[ds['Country'] == 'India']
ds = ds[ds['State'] != '']

#Plot Map
qtd = ds['State'].value_counts()
calor = ds[['LONGITUDE', 'LATITUDE', 'TARGET(PRICE_IN_LACS)']]

mapa = folium.Map(width="100%",height="100%", location = [20.59403,78.96290], zoom_start=4)
mapa = mapa.add_child(plugins.HeatMap(calor))
mapa.save("hot-map.html")

#Select columns
X = ds[['UNDER_CONSTRUCTION', 'RERA', 
        'SQUARE_FT', 'READY_TO_MOVE', 'RESALE',
       'Builder', 'Dealer', 'CITY', 'NEIGHBORHOOD',
       'Owner', 'LONGITUDE', 'LATITUDE']]

y = ds['TARGET(PRICE_IN_LACS)']

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

#Append Models
models = []

model = ExtraTreeRegressor()
models.append(model)

model = DecisionTreeRegressor()
models.append(model)

model = LinearRegression(n_jobs=-1)
models.append(model)

model = xgboost.XGBRegressor(n_jobs=-1)
models.append(model)

model = lgbm.LGBMRegressor(verbose=1)
models.append(model)

#Fit, Predict and Metrics
name_models = []
error = []

for model in models:
    name_models.append(type(model).__name__)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    error.append(mean_absolute_error(y_test, y_predict))


#Plot error
plt.bar(name_models, error, color='blue')
plt.xticks(name_models)
plt.ylabel('Mean absolute error')
plt.xlabel('Models')
plt.title('Results')
plt.show()

#Feature engineering
xgb = xgboost.XGBRegressor(n_jobs=-1)

select = RFE(xgb, 10)
model = select.fit(X,y)
features = model.transform(X)
colums = model.get_support(indices=True)
X = X.iloc[:,colums]

#Grid Search
param_grid = {
        'min_child_weight': [1, 5, 10, None],
        'gamma': [0.5, 1, 1.5, 2, 5, None],
        'subsample': [0.6, 0.8, 1.0, None],
        'colsample_bytree': [0.6, 0.8, 1.0, None],
        'max_depth': [3, 4, 5, None]
        }

grid = GridSearchCV(xgb, param_grid=param_grid, cv=10, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)
y_predict = grid.predict(X_test)
erro = mean_absolute_error(y_test, y_predict)