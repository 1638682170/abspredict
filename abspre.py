#导入库
from rdkit import Chem
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV, cross_val_score, cross_validate
import os
import glob
from scipy.stats import norm
import math
import random
import streamlit as st
import time
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
plt.style.use('ggplot')
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import joblib


#界面设计
#数据上传与侧栏
st.sidebar.subheader('* 数据上传 *')
st.subheader('*模型训练 *')
fileX = st.sidebar.file_uploader('上传指纹特征数据', type=['xlsx'], key=None)
if fileX is not None:
    dataX = pd.read_excel(fileX)
    st.write(dataX)
    X=np.array(dataX)
else:
    st.write('请上传数据')


#处理随机森林中nan问题 try
for i in range(len(X)):
                    sample= X[i]
                    for j in range(len(sample)):
                        if np.isnan(sample[j]):
                            sample[j]=0 
#等待条
with st.spinner("Wait for it..."): 
    for i in range(100):
        time.sleep(0.05)

st.success("Done!")


#各个机器学习模型参数

LGBM = lgb.LGBMRegressor(n_estimators=540)
lgb.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       importance_type='split', learning_rate=0.1, max_depth=-1,feature_fraction=0.6,bagging_fraction=0.6,
       min_child_samples=10, min_child_weight=0.1, min_split_gain=0.0,
       n_estimators=540, n_jobs=-1, objective='regression', num_leaves=30,
       random_state=None, reg_alpha=0.6, reg_lambda=0.0, silent=True,
       subsample=0.6, subsample_for_bin=200000, subsample_freq=0)
option = st.selectbox('select your goal',('absorption wavelength', 'abs FWHM'))
st.write('You selected:', option)
loaded_model1 = joblib.load('LGBM_absw.dat')
loaded_model2 = joblib.load('LGBM_absf.dat')
if option=='absorption wavelength':
	y_pred_train=loaded_model1.predict(X)
else:
	y_pred_train=loaded_model2.predict(X)



my_large_df=pd.DataFrame(y_pred_train)

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(my_large_df)

st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name='large_df.csv',
     mime='text/csv',
 )

