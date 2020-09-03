# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:27:40 2020

@author: Jinhong Chan

email：842960911@qq.com
"""

import lightgbm as lgb
import feature
import os
from sklearn.externals import joblib


res_path = "../res/"

def gen_res(res_file = res_path + "prediction.csv", model_file = res_path + "model.pkl"):
    # 载入数据
    train, test = feature.feat_merger()
    train_X = train.drop(columns=["user_id","merchant_id","label"])
    train_y = train["label"]
    test_X = test.drop(columns=["user_id","merchant_id","label"])
    if os.path.exist(model_file):
        model = joblib.load(model_file)
    else:
        # 模型设定
        model = lgb.LGBMClassifier(bagging_fraction=0.6, bagging_freq=0, boosting_type='gbdt',
                    class_weight=None, colsample_bytree=1.0, feature_fraction=0.7,
                    importance_type='split', lambda_l1=0.9, lambda_l2=0.5,
                    learning_rate=0.01, max_bin=245, max_depth=5, metrics='auc',
                    min_child_samples=20, min_child_weight=0.001,
                    min_data_in_leaf=261, min_split_gain=0.4, n_estimators=2000,
                    n_jobs=-1, num_leaves=15, objective='binary', random_state=None,
                    reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                    subsample_for_bin=200000, subsample_freq=0)
        # 模型训练
        model.fit(train_X, train_y)
        # 模型保存
        joblib.dump(model, model_file)
    # 生成结果
    res = test[["user_id","merchant_id"]]
    res["prob"] = model.predict_proba(test_X)[:,1]
    res.to_csv(res_file, index=False, header=True)
    return model, res
