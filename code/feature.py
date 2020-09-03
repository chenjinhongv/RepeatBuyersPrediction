# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:27:27 2020

@author: Jinhong Chan

email：842960911@qq.com
"""


import pandas as pd
import os

BASE_PATH = "../data/"


def load_ori_data():
    """

    Returns
    -------
    info : pandas.Dataframe
        A pandas.Dataframe for user base info with columns:
            user_id
            age_range
            gender
    actions : pandas.Dataframe
        A pandas.Dataframe for user action in merchant with columns:
            user_id
            merchant_id
            item_id：物品id
            cat_id:品类id
            brand_id：品牌id
            time_stamp：实践戳，脱敏
            action_type:0：点击 1：加购物车 2：购买 3：加收藏
    userXmerchant : pandas.Dataframe
        A pandas.Dataframe to sign if user is a repeatbuyer for merchant with columns:
            user_id
            merchant_id
            label:0:not repeatbuyer 1：repeatbuyer -1：unhnow object to be predict

    """

    info = pd.read_csv(BASE_PATH + "user_info_format1.csv")
    
    actions = pd.read_csv(BASE_PATH + "user_log_format1.csv")
    actions.rename(columns={"seller_id":"merchant_id"},inplace=True)
    
    userXmerchant_with_label = pd.read_csv(BASE_PATH + "train_format1.csv")
    userXmerchant_for_submit = pd.read_csv(BASE_PATH + "test_format1.csv")
    userXmerchant = pd.concat([userXmerchant_with_label,userXmerchant_for_submit])
    userXmerchant["label"].fillna(-1,inplace=True)
    
    return info, actions, userXmerchant


def user_action_feat(outfile = BASE_PATH + "user_action_feat.csv"):
    """

    Returns
    -------
    user_action_feat : pandas.Dataframe
        A pandas.Dataframe to describe user actions feature with columns:
            user_id
            user_actions_count:用户行为计数特征
            user_action_type0_count:用户点击行为计数
            user_action_type1_count:用户加购物车行为计数
            user_action_type2_count:用户购买行为计数
            user_action_type3_count:用户加收藏行为计数
            user_action_nopurchase_count：非购买行为计数
            user_purchase_rate：购买行为比例
            user_merchant_count：用户有行为的商家计数
    """

    if os.path.exists(outfile):
        return pd.read_csv(outfile)
    else:
        info, actions, userXmerchant = load_ori_data()

        # 初始化user_action_feat
        user_action_feat = userXmerchant[["user_id"]].drop_duplicates(subset=["user_id"])

        # 用户行为计数特征
        user_actions_count = actions[["user_id","action_type"]].groupby(["user_id"]).count().reset_index()
        user_actions_count.rename(columns={"action_type":"user_actions_count"},inplace=True)
        user_action_feat = pd.merge(user_action_feat,user_actions_count,on="user_id",how="left")
        user_action_feat["user_actions_count"].fillna(0,inplace=True)

        # 用户各种行为计数
        # 0：点击 1：加购物车 2：购买 3：加收藏
        for i in [0, 1, 2, 3]:
            temp = actions.loc[actions.action_type == i,["user_id","action_type"]].groupby(["user_id"]).count().reset_index()
            temp.rename(columns={"action_type":"user_action_type" + str(i) + "_count"},inplace=True)
            user_action_feat = pd.merge(user_action_feat,temp,on="user_id",how="left")
            user_action_feat["user_action_type" + str(i) + "_count"].fillna(0,inplace=True)

        # 非购买行为计数
        temp = actions.loc[actions.action_type != 2,["user_id","action_type"]].groupby(["user_id"]).count().reset_index()
        temp.rename(columns={"action_type":"user_action_nopurchase_count"},inplace=True)
        user_action_feat = pd.merge(user_action_feat,temp,on="user_id",how="left")
        user_action_feat["user_action_nopurchase_count"].fillna(0,inplace=True)

        # 用户购买行为比例
        user_action_feat["user_purchase_rate"] = user_action_feat["user_action_type2_count"]/user_action_feat["user_actions_count"]
        user_action_feat["user_purchase_rate"].fillna(0,inplace=True)

        # 用户有行为的店铺计数
        user_merchant_count = actions[["user_id","merchant_id"]].drop_duplicates(subset=["user_id","merchant_id"]).groupby(["user_id"]).count().reset_index()
        user_merchant_count.rename(columns={"merchant_id":"user_merchant_count"},inplace=True)
        user_action_feat = pd.merge(user_action_feat,user_merchant_count,on="user_id",how="left")
        user_action_feat["user_merchant_count"].fillna(0,inplace=True)

        user_action_feat.to_csv(outfile, index=False, header=True)
        return user_action_feat


def userXmerchant_action_feat(outfile = BASE_PATH + "userXmerchant_action_feat.csv"):
    """

    Parameters
    ----------
    outfile : str, optional
        DESCRIPTION. The default is BASE_PATH + "userXmerchant_action_feat.csv".

    Returns
    -------
    userXmerchant_action_feat : pandas.Dataframe
        A pandas.Dataframe to describe userXmerchant actions feature with columns:
            user_id
            merchant_id
            userXmerchant_actions_count：用户x店铺的行为计数
            userXmerchant_action_type0_count：用户x店铺的点击行为计数
            userXmerchant_action_type1_count：用户x店铺的加购物车行为计数
            userXmerchant_action_type2_count：用户x店铺的购买行为计数
            userXmerchant_action_type3_count：用户x店铺的收藏行为计数
            userXmerchant_purchase_rate：用户x店铺购买行为比例
            userXmerchant_actions_fraction：用户对当前店铺的行为占其行为总数的比例

    """
    
    if os.path.exists(outfile):
        return pd.read_csv(outfile)
    else:
        info, actions, userXmerchant = load_ori_data()
        
        # 初始化userXmerchant_action_feat
        userXmerchant_action_feat = userXmerchant[["user_id","merchant_id"]]
        
        # 过滤无关行为数据
        actions_TuserXmerchant = pd.merge(actions,userXmerchant,on=["user_id","merchant_id"],how="inner")
        
        # 用户x店铺行为计数特征
        userXmerchant_actions_count = actions_TuserXmerchant[["user_id","merchant_id","action_type"]].groupby(["user_id","merchant_id"]).count().reset_index()
        userXmerchant_actions_count.rename(columns={"action_type":"userXmerchant_actions_count"},inplace=True)
        userXmerchant_action_feat = pd.merge(userXmerchant_action_feat,userXmerchant_actions_count,on=["user_id","merchant_id"],how="left")
        userXmerchant_action_feat["userXmerchant_actions_count"].fillna(0,inplace=True)
        
        # 用户x店铺各种行为计数
        for i in [0, 1, 2, 3]:
            temp = actions_TuserXmerchant.loc[actions_TuserXmerchant.action_type == i,["user_id","merchant_id","action_type"]].groupby(["user_id","merchant_id"]).count().reset_index()
            temp.rename(columns={"action_type":"userXmerchant_action_type" + str(i) + "_count"},inplace=True)
            userXmerchant_action_feat = pd.merge(userXmerchant_action_feat,temp,on=["user_id","merchant_id"],how="left")
            userXmerchant_action_feat["userXmerchant_action_type" + str(i) + "_count"].fillna(0,inplace=True)
        
        # 用户x店铺购买行为比例
        userXmerchant_action_feat["userXmerchant_purchase_rate"] = userXmerchant_action_feat["userXmerchant_action_type2_count"]/userXmerchant_action_feat["userXmerchant_actions_count"]
        userXmerchant_action_feat["userXmerchant_purchase_rate"].fillna(0,inplace=True)
        
        # 用户对当前商家的行为占其行为总数的比例
        user_actions_count = actions[["user_id","action_type"]].groupby(["user_id"]).count().reset_index()
        user_actions_count.rename(columns={"action_type":"user_actions_count"},inplace=True)
        userXmerchant_action_feat = pd.merge(userXmerchant_action_feat,user_actions_count,on="user_id",how="left")
        userXmerchant_action_feat["user_actions_count"].fillna(0,inplace=True)
        userXmerchant_action_feat["userXmerchant_actions_fraction"] = userXmerchant_action_feat["userXmerchant_actions_count"]/userXmerchant_action_feat["user_actions_count"]
        userXmerchant_action_feat["userXmerchant_actions_fraction"].fillna(0,inplace=True)
        del userXmerchant_action_feat["user_actions_count"]
        
        userXmerchant_action_feat.to_csv(outfile, index=False, header=True)
        return userXmerchant_action_feat

def merchant_action_feat(outfile = BASE_PATH + "merchant_action_feat.csv"):
    """

    Parameters
    ----------
    outfile : str, optional
        DESCRIPTION. The default is BASE_PATH + "merchant_action_feat.csv".

    Returns
    -------
    merchant_action_feat : pandas.Dataframe
        A pandas.Dataframe to describe merchant actions feature with columns:
            merchant_id
            merchant_act_count:店铺顾客行为总数
            merchant_purchase_rate：商家顾客购买行为比例
            merchant_user_count：商家顾客数量
            merchant_purchase_user_count：商家发生购买行为的客户数量

    """
    
    if os.path.exists(outfile):
        return pd.read_csv(outfile)
    else:
        info, actions, userXmerchant = load_ori_data()
        
        # 初始化merchant_action_feat
        merchant_action_feat = userXmerchant["merchant_id"].drop_duplicates(subset=["merchant_id"])
        
        # 店铺顾客行为总数
        merchant_act_count = actions[["merchant_id","action_type"]].groupby(["merchant_id"]).count().reset_index()
        merchant_act_count.rename(columns={"action_type":"merchant_act_count"},inplace=True)
        merchant_action_feat = pd.merge(merchant_action_feat,merchant_act_count,on="merchant_id",how="left")
        merchant_action_feat["merchant_act_count"].fillna(0,inplace=True)
        
        # 店铺购买行为总数
        merchant_purchase_rate = actions.loc[actions.action_type == 2,["merchant_id","action_type"]].groupby(["merchant_id"]).count().reset_index()
        merchant_purchase_rate.rename(columns={"action_type":"merchant_purchase_rate"},inplace=True)
        merchant_action_feat = pd.merge(merchant_action_feat,merchant_purchase_rate,on="merchant_id",how="left")
        merchant_action_feat["merchant_purchase_rate"].fillna(0,inplace=True)
        
        # 店铺客流量
        merchant_user_count = actions[["merchant_id","user_id"]].drop_duplicates(["merchant_id","user_id"]).groupby(["merchant_id"]).count().reset_index()
        merchant_user_count.rename(columns={"user_id":"merchant_user_count"},inplace=True)
        merchant_action_feat = pd.merge(merchant_action_feat,merchant_user_count,on="merchant_id",how="left")
        merchant_action_feat["merchant_user_count"].fillna(0,inplace=True)
        
        # 店铺客户转化数
        merchant_purchase_user_count = actions.loc[actions.action_type == 2,["merchant_id","user_id"]].drop_duplicates(["merchant_id","user_id"]).groupby(["merchant_id"]).count().reset_index()
        merchant_purchase_user_count.rename(columns={"user_id":"merchant_purchase_user_count"},inplace=True)
        merchant_action_feat = pd.merge(merchant_action_feat,merchant_purchase_user_count,on="merchant_id",how="left")
        merchant_action_feat["merchant_purchase_user_count"].fillna(0,inplace=True)
        
        merchant_action_feat.to_csv(outfile, index=False, header=True)
        return merchant_action_feat

def feat_merger():
    
    # merge
    info, actions, userXmerchant = load_ori_data()
    temp = pd.merge(userXmerchant, info, on=["user_id"], how="left")
    temp = pd.merge(temp, userXmerchant_action_feat(), on=["user_id","merchant_id"], how="left")
    temp = pd.merge(temp, user_action_feat(), on=["user_id"], how="left")
    temp = pd.merge(temp, merchant_action_feat(), on=["merchant_id"], how="left")
    
    # train&test split
    train = temp.loc[temp.label != -1,:]
    test = temp.loc[temp.label == -1,:]
    return train, test