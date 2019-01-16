 # Standard libraries
#import os
import numpy as np
import pandas as pd

# Visualization
#import matplotlib.pyplot as plt
#import seaborn as sns
#from pdpbox import pdp
#from plotnine import *
#from pandas_summary import DataFrameSummary
#from sklearn.ensemble import RandomForestRegressor
#from IPython.display import display

# Machine Learning
#import sklearn
#from sklearn import metrics
#from scipy.cluster import hierarchy as hc
# from fastai.imports import *

#%%
# Import dataset
dtypes = {
        'Id'                : 'object',
        'groupId'           : 'object',
        'matchId'           : 'object',
        'assists'           : 'int8',
        'boosts'            : 'int8',
        'damageDealt'       : 'float16',
        'DBNOs'             : 'int8',
        'headshotKills'     : 'int8', 
        'heals'             : 'int8',    
        'killPlace'         : 'int8',    
        'killPoints'        : 'int16',    
        'kills'             : 'int8',    
        'killStreaks'       : 'int8',    
        'longestKill'       : 'float16',
        'matchDuration'     : 'int16',
        'matchType'         : 'object',
        'maxPlace'          : 'int8',    
        'numGroups'         : 'int8',
        'rankPoints'        : 'int16',
        'revives'           : 'int8',    
        'rideDistance'      : 'float16',    
        'roadKills'         : 'int8',    
        'swimDistance'      : 'float16',    
        'teamKills'         : 'int8',    
        'vehicleDestroys'   : 'int8',    
        'walkDistance'      : 'float16',    
        'weaponsAcquired'   : 'int8',    
        'winPoints'         : 'int8', 
        'winPlacePerc'      : 'float16' 
}
#%%
train = pd.read_csv('data/train_V2.csv', nrows = 1000000, dtype = dtypes)
#test = pd.read_csv('data/test_V2.csv')
train.info()

#%%
del train
import gc
gc.collect()
#%% Initial Exploration

train.head()
#test.head()

train.describe()

# Types, Data points, memory usage, etc.
train.info()

#%% Illegal Match
# Fellow Kaggler 'averagemn' brought to our attention that there is one particular player with a 'winPlacePerc' of NaN. The case was that this match had only one player. We will delete this row from our dataset.

# Check row with NaN value
train[train['winPlacePerc'].isnull()]

# Drop row with NaN 'winPlacePerc' value
train.drop(2744604, inplace=True)

# The row at index 2744604 will be gone
train[train['winPlacePerc'].isnull()]

#%% Feature Engineering

# totalFeatures
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']

# killsWithoutMoving
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

# headshot_rate
train['headshot_rate'] = train['headshotKills'] + train['kills']
train['headshot_rate'] = train['headshotKills'].fillna(0) # check number of missing values before imputing

# healsandboosts
train['healsandboosts'] = train['heals'] + train['boosts']

# playersJoined
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')


 #%% OUTLIERS

# Players who kill without moving 
#train[train['killsWithoutMoving'] == True].shape
# Remove them 
train.drop(train[train['killsWithoutMoving'] == True].index, inplace = True)
#######################

# Players with more than 10 roadKills
train[train['roadKills'] > 10]
# Remove them
train.drop(train[train['roadKills'] > 10].index, inplace = True)
#######################

# Players with more than 30 kills
#train[train['kills'] > 30].shape
# Remove them 
train.drop(train[train['kills'] > 30].index, inplace = True)
#######################

# Players who made kills from a diatance greater than 1 KM
#train[train['longestKill'] >= 1000].shape
# Remove them 
train.drop(train[train['longestKill'] > 30].index, inplace = True)
#######################

# DISTANCE

# Players who made walked greater than 10000
#train[train['walkDistance'] >= 10000].shape
# Remove them 
train.drop(train[train['walkDistance'] > 10000].index, inplace = True)
#######################

# Players who rode greater than 20000
#train[train['rideDistance'] >= 20000].shape
# Remove them 
train.drop(train[train['rideDistance'] > 20000].index, inplace = True)
#######################

# Players who swam greater than 2000
#train[train['swimDistance'] >= 2000].shape
# Remove them 
train.drop(train[train['swimDistance'] > 2000].index, inplace = True)
#######################

# SUPPLIES
# Players with more than 80 weapons acquired
#train[train['weaponsAcquired'] >= 80].shape
# Remove them
train.drop(train[train['weaponsAcquired'] > 80].index, inplace = True)
#######################

# Players with more than 40 heals used
#train[train['heals'] >= 40].shape
# Remove them
train.drop(train[train['heals'] > 40].index, inplace = True)
#######################

# After all the outlier removal, 2000 players have been removed. Check.
# 207116 rows removed 
#######################################################################

#%% Categorical Variables

# Number of different matchTypes 
train['matchType'].nunique() 

# turn groupId and matchId into categorical types
train['groupId'] = train['groupId'].astype('category')
train['matchId'] = train['matchId'].astype('category')

# get category codes for groupId and matchId
train['groupId_cat'] = train['groupId'].cat.codes
train['matchId_cat'] = train['matchId'].cat.codes

train.drop(['groupId','matchId'], axis = 1, inplace = True)
# train.drop(columns = ['groupId','matchId'], inplace = True)

train.drop(columns = ['Id'], inplace = True)


#%% Label Encode matchType column

train = pd.get_dummies(train, columns = ['matchType', 'killsWithoutMoving'], drop_first = True)


#%%


y = train['winPlacePerc']
train.drop(['winPlacePerc'], axis = 1, inplace = True)
#%% TRAIN TEST SPLITTING
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.1, random_state = 0)

#%% Modeling
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)

y_pred = rfr.predict(X_test)

#%% Evaluation Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
import math

print("MAE is",mean_absolute_error(y_test, y_pred))
print("100xRMSLE is",100*math.sqrt(mean_squared_log_error(y_test, y_pred)))

#%% Transform on test set

test['headshot_rate'] = test['headshotKills'] / test['kills']
test['headshot_rate'] = test['headshot_rate'].fillna(0)
test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']
test['playersJoined'] = test.groupby('matchId')['matchId'].transform('count')
test['killsNorm'] = test['kills']*((100-test['playersJoined'])/100 + 1)
test['damageDealtNorm'] = test['damageDealt']*((100-test['playersJoined'])/100 + 1)
test['maxPlaceNorm'] = test['maxPlace']*((100-train['playersJoined'])/100 + 1)
test['matchDurationNorm'] = test['matchDuration']*((100-test['playersJoined'])/100 + 1)
test['healsandboosts'] = test['heals'] + test['boosts']
test['killsWithoutMoving'] = ((test['kills'] > 0) & (test['totalDistance'] == 0))

# Turn groupId and match Id into categorical types
test['groupId'] = test['groupId'].astype('category')
test['matchId'] = test['matchId'].astype('category')

# Get category coding for groupId and matchID
test['groupId_cat'] = test['groupId'].cat.codes
test['matchId_cat'] = test['matchId'].cat.codes

# Remove irrelevant features from the test set
test_pred = test[to_keep].copy()

# Fill NaN with 0 (temporary)
test_pred.fillna(0, inplace=True)
test_pred.head()

#%% Predict and Export to CSV
submission = np.clip(a = rfr.predict(test), a_min = 0.0, a_max = 1.0)
submission_df = pd.DataFrame({'Id': test['Id'], 'winPlacePerc': submission})

submission_df.to_csv("v1.csv", index = False, header = True)