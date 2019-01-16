# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_dtypes = {
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

test_dtypes = {
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
}
print("\n==================Step 1: Importing datasets ===========================")

train = pd.read_csv('../input/train_V2.csv', dtype = train_dtypes)
test = pd.read_csv('../input/test_V2.csv', dtype = test_dtypes)

test_df_ID = test['Id']

#%% Illegal Match
# Fellow Kaggler 'averagemn' brought to our attention that there is one particular player with a 'winPlacePerc' of NaN. The case was that this match had only one player. We will delete this row from our dataset.

# Check row with NaN value
train[train['winPlacePerc'].isnull()]

# Drop row with NaN 'winPlacePerc' value
train.drop(2744604, inplace=True)

# The row at index 2744604 will be gone
train[train['winPlacePerc'].isnull()]

#%% Concat train and test
y = train['winPlacePerc']
train.drop(['winPlacePerc'], axis = 1, inplace = True)
df = pd.concat([train,test], axis = 0)

#%%
# save the len of train and test  (number of rows)
len_train = train.shape[0]
#len_test = test.shape[0]
# add garbage collect code here to remove train and test

import gc
del train, test
gc.collect()


#%% Feature Engineering
print("\n==================Step 2: Feature Engineering ==========================")

# totalFeatures
df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']

# killsWithoutMoving
#df['killsWithoutMoving'] = ((df['kills'] > 0) & (df['totalDistance'] == 0))

# headshot_rate
df['headshot_rate'] = df['headshotKills'] + df['kills']
df['headshot_rate'] = df['headshotKills'].fillna(0) # check number of missing values before imputing

# healsandboosts
df['healsandboosts'] = df['heals'] + df['boosts']

# playersJoined
df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')

#%%
def calc_killsWithoutMoving(x,y):
        if x > 0 & int(y) == 0:
                return 1
        else :
                return 0

df['killsWithoutMoving'] = df[['kills','totalDistance']].apply(lambda x: calc_killsWithoutMoving(*x), axis=1)

#%% Categorical Variables
# Number of different matchTypes 
#df['matchType'].nunique() 

# turn groupId and matchId into categorical types
df['groupId'] = df['groupId'].astype('category')
df['matchId'] = df['matchId'].astype('category')

# get category codes for groupId and matchId
df['groupId_cat'] = df['groupId'].cat.codes
df['matchId_cat'] = df['matchId'].cat.codes

df.drop(['groupId','matchId'], axis = 1, inplace = True)
# df.drop(columns = ['groupId','matchId'], inplace = True)

df.drop(columns = ['Id'], inplace = True)

# Label Encode matchType column
df = pd.get_dummies(df, columns = ['matchType'], drop_first = True)

#%% Split train and test df

print("\n==============Step 3: Split train and submission DFs====================")
train_df = df.iloc[:len_train, :]
test_df = df.iloc[len_train:, :]
train_df = pd.concat([train_df,y],axis = 1)

#%% Outlier removal on train 
# concat y then remove outliers
# print("\n==================Step : ==========================")

#%% OUTLIERS
# Players who kill without moving 
#train_df[train_df['killsWithoutMoving'] == True].shape
# Remove them 
train_df.drop(train_df[train_df['killsWithoutMoving'] == True].index, inplace = True)
#######################

# Players with more than 10 roadKills
# train_df[train_df['roadKills'] > 10]
# Remove them
train_df.drop(train_df[train_df['roadKills'] > 10].index, inplace = True)
#######################

# Players with more than 30 kills
#train_df[train_df['kills'] > 30].shape
# Remove them 
train_df.drop(train_df[train_df['kills'] > 30].index, inplace = True)
#######################

# Players who made kills from a distance greater than 1 KM
#train_df[train_df['longestKill'] >= 1000].shape
# Remove them 
train_df.drop(train_df[train_df['longestKill'] > 1000].index, inplace = True)
#######################

# DISTANCE

# Players who made walked greater than 10000
#train_df[train_df['walkDistance'] >= 10000].shape
# Remove them 
train_df.drop(train_df[train_df['walkDistance'] > 10000].index, inplace = True)
#######################

# Players who rode greater than 20000
#train_df[train_df['rideDistance'] >= 20000].shape
# Remove them 
train_df.drop(train_df[train_df['rideDistance'] > 20000].index, inplace = True)
#######################

# Players who swam greater than 2000
#train_df[train_df['swimDistance'] >= 2000].shape
# Remove them 
train_df.drop(train_df[train_df['swimDistance'] > 2000].index, inplace = True)
#######################

# SUPPLIES
# Players with more than 80 weapons acquired
#train_df[train_df['weaponsAcquired'] >= 80].shape
# Remove them
train_df.drop(train_df[train_df['weaponsAcquired'] > 80].index, inplace = True)
#######################

# Players with more than 40 heals used
#train_df[train_df['heals'] >= 40].shape
# Remove them
train_df.drop(train_df[train_df['heals'] > 40].index, inplace = True)
#######################

# After all the outlier removal, 2000 players have been removed. Check.
# 207116 rows removed 
y = train_df['winPlacePerc']

#%% Train Test Split on train df
print("\n===================Step 4: Train Test Split ===========================")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:,:-1], y, test_size = 0.2, random_state = 0)

#%% Modeling
print("\n=====================Step 5: Modeling =================================")
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)

y_pred = rfr.predict(X_test)

#%% Evaluation Metrics
print("\n==================Step 6: Evaluation Metrics ==========================")
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_log_error
# import math

# print("\nMAE is",mean_absolute_error(y_test, y_pred))
# print("\nRMSE is",math.sqrt(mean_squared_log_error(y_test, y_pred)))

#%% Predict on test_df
print("\n==================Step 7: Predict and Export to CSV ===================")
submission = np.clip(a = rfr.predict(test_df), a_min = 0.0, a_max = 1.0)

#%% Export to CSV
submission_df = pd.DataFrame({'Id': test_df_ID, 'winPlacePerc': submission})

submission_df.to_csv("v5.csv", index = False, header = True)
print("\n================== Done! ==========================")