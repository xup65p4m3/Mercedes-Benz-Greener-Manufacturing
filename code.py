import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

######################################################
train = pd.read_csv('data/train.csv')
train = shuffle(train[train.y < 250], random_state=0).reset_index(drop=True)
test = pd.read_csv('data/test.csv')
test_ID = test.ID.values

train_cat_X = train.drop(columns=['ID', 'y'])
train_cat_Y = train.y.values
test_cat_X = test.drop(columns=['ID'])

'''label encoding'''
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

train_X = train.drop(columns=['ID', 'y'])
train_Y = train.y.values
test_X = test.drop(columns=['ID'])

######################################################
'''Catboost'''
cat = CatBoostRegressor(
iterations=1000,
learning_rate=0.05,
depth=4)

cat_features = list(range(9))
cat.fit(train_cat_X, train.y.values, cat_features)
cat_cv_score = cross_val_score(estimator=cat, X=train_cat_X, y=train_cat_Y, fit_params={'cat_features':cat_features}, scoring='r2', cv=10)
print(cat_cv_score.mean())

cat_train_pred = cat.predict(train_cat_X)
cat_test_pred = cat.predict(test_cat_X)

######################################################
'''Xgboost'''
xgb = XGBRegressor(
max_depth = 4,
learning_rate = 0.06,
n_estimators = 100,
subsample = 0.95,
colsample_bytree = 0.95)

xgb.fit(train_X, train_Y)
xgb_cv_score = cross_val_score(estimator=xgb, X=train_X, y=train_Y, scoring='r2', cv=10)
print(xgb_cv_score.mean())

xgb_train_pred = xgb.predict(train_X)
xgb_test_pred = xgb.predict(test_X)

######################################################
'''Lightgbm'''
lgb = LGBMRegressor(
num_leaves=10,
max_depth=4,
learning_rate=0.005,
n_estimators=900,
subsample=0.995,
subsample_freq=1)

lgb.fit(train_X, train_Y)
lgb_cv_score = cross_val_score(estimator=lgb, X=train_X, y=train_Y, scoring='r2', cv=10)
print(lgb_cv_score.mean())

lgb_train_pred = lgb.predict(train_X)
lgb_test_pred = lgb.predict(test_X)

######################################################
'''Random Forest'''
rf = RandomForestRegressor(n_estimators=200, max_depth=6)
rf.fit(train_X, train_Y)
rf_cv_score = cross_val_score(estimator=rf, X=train_X, y=train_Y, scoring='r2', cv=10)
print(rf_cv_score.mean())

rf_train_pred = rf.predict(train_X)
rf_test_pred = rf.predict(test_X)

######################################################
final_pred = (1/4)*cat_test_pred + (1/4)*xgb_test_pred + (1/4)*lgb_test_pred + (1/4)*rf_test_pred
result_table = pd.DataFrame({'ID':test_ID, 'y': final_pred})
result_table.to_csv('result.csv', index=False)
