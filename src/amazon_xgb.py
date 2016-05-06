import numpy as np
import scipy.sparse
import pickle
import pandas as pd
import xgboost as xgb
import numpy as np
import scipy.sparse
import pickle
import pandas as pd
import xgboost as xgb

dtrain = xgb.DMatrix('/Users/admin/Dropbox/EE379K/project/ee379k_project/dataset/train.csv')

param = {'max_depth':2, 'eta':0.5, 'silent':1, 'objective':'binary:logistic'}

num_round = 300

param = {'max_depth':100,
    'silent':1, 'objective':'binary:logistic',
        'max_delta_step':1.5, 'n_estimator':200, 'colsample_bytree':0.3,
            'scale_pos_weight':0.0614, }

#load prev results
predictions_logis = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submission_1462240590.csv', header = 0)
predictions_logis = predictions_logis['ACTION']
predictions_fm = pd.read_csv('/Users/admin/Dropbox/EE379K/project/ee379k_project/submissions/submission_FM.csv', header = 0)
predictions_fm = predictions_fm['ACTION']

predictions1 = pd.read_csv('pred1.csv', header = 0)
predictions1 = predictions1['ACTION']
predictions2 = pd.read_csv('pred2.csv', header = 0)
predictions2 = predictions2['ACTION']

#Tune the coef
predictions = 0.44 * (0.35 * predictions1 + 0.15 * predictions2 + 0.5 * predictions_logis) + 0.56 * predictions_fm

print(predictions)
test_df = pd.read_csv('test.csv', header = 0)
submission = pd.DataFrame({'id':test_df['id'], 'ACTION':predictions})
submission.to_csv('submission_xgb.csv', index = False)

