import numpy as np
import pandas as pd

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

train = pd.read_csv('data/train_V2.csv', nrows = 10000, dtype = train_dtypes)
test = pd.read_csv('data/test_V2.csv', dtype = test_dtypes)

test_df_ID = test['Id']
#%%

import gc
del train, test
gc.collect()


#%% Concat train and test

y = train['winPlacePerc']
train.drop(['winPlacePerc'], axis = 1, inplace = True)
df = pd.concat([train,test], axis = 0)

#%%
# save the len of train and test  (number of rows)

len_train = train.shape[0]
#len_test = test.shape[0]
# add garbage collect code here to remove train and test

#%% Feature Engineering

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

train_df = df.iloc[:len_train, :]
test_df = df.iloc[len_train:, :]
train_df = pd.concat([train_df,y],axis = 1)

#%% Outlier removal on train 
# concat y then remove outliers

#%% Train Test Split on train df

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:,:-1], y, test_size = 0.2, random_state = 0)

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

#%% Predict on test_df

submission = np.clip(a = rfr.predict(test_df), a_min = 0.0, a_max = 1.0)

#%% Export to CSV

submission_df = pd.DataFrame({'Id': test_df_ID, 'winPlacePerc': submission})

submission_df.to_csv("submissions/v1.csv", index = False, header = True)