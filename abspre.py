#导入库
import pandas as pd
import numpy as np
import os
import glob
import time
import math
import random
import streamlit as st
import time
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
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


#选择机器学习模型
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

