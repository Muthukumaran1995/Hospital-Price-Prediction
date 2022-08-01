# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:23:52 2021

@author: jayac
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor,BaggingRegressor
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
from sklearn.metrics import  r2_score 
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor



#-----------------------#
# Page Layout
st.set_page_config(layout="wide")

st.title('Aggregation Run Time Predictor')
image = Image.open(r'C:\Users\jayac\Downloads\Dataset\CIP\NielsenIQ-logo_1603989929.jpg')

st.image(image, width =600)



st.markdown ("""
            This app predicts the run time of Aggregation for each FDS
        
             """)

#About

expander_bar = st.beta_expander("INPUT FIELDS Details")
expander_bar.markdown("""
* **FACTS**	Number of Facts
* **MARKET HIERARCHY** 	Number Of Market Hierarchy 
* **MARKET** 	Number of Markets:
* **PRODUCT HIERARCHY**	Number of Product Hierarchy:
* **PRODUCT NODE**	Number of Product Nodes
* **ITEM**	Number of Items in FDS
* **WEEK	**  Number of Weeks
* **ITEM LEVEL FLAG**	Item Level Flag Enabled for FDS or Not
* **PROMOTIONS** 	Number of Promotions
* **CHANNEL_COUNT**	 Number of Channels linked to FDS
* **SXS FLAG**	Shop Level Flag in FDS
* **MARKET_IS_COMPOSITE**	Composite Markets linked in FDS
* **PEAK PRODUCTION**	 FDS PRODUCED IN PEAK PRODUCTION
* **THI_PER_PERIODICITY** 	FDS is Calendar or Weekly Periodicity
* **QUEUE_NAME** 	FDS running in Large or Small Queue

                      """)
#---------------------------------------------------#

#Page Layout

col1 = st.sidebar
col2, col3 = st.beta_columns((1,1))


#---------------------------------------------#

col1.header('Input Options')

## Selectbox

sel_facts = col1.slider('Select Number of Facts', 1, 100, 20)
sel_mkt_hie = col1.slider('Select Number Of Market Hierarchy', 1, 1000, 20) 
sel_mkt = col1.slider('Select Number of Markets', 1, 1000,200)
sel_prd_hie = col1.slider('Select Number of Product Hierarchy', 1, 1000,200)
sel_prd_node = col1.slider('Select Number of Product Node', 1, 1000000,5000)
sel_items = col1.slider('Select Number of Items', 1, 1500000,50000)
sel_week = col1.slider('Select Number of Weeks', 1, 400,175)
sel_itm_flag = col1.radio( "Choose Item Level Flag", 
                         ('Yes','No'))
sel_fct_prd = col1.slider('Select Number of Promotions', 1, 1000,100)
sel_cch = col1.slider('Select Number of Channels', 1, 20,2)
sel_sxs = col1.radio( "Choose SXS Flag", ('Yes','No'))
sel_cmp = col1.radio( "Choose Market is Composite", ('Yes','No'))
sel_peak_prod = col1.radio( "Choose Peak Production", ('Yes','No'))
sel_periodicity = col1.radio( "Choose Peak Production", ('WEEKLY','CALENDAR'))
sel_queue = col1.radio( "Choose Peak Production", ('Large','Small'))
sel_alg = col1.radio( "Choose Your Algorithm for Prediction", 
                         (('RandomForest','XGBoost','LightGBM','Voting Regressor','Stacking Regressor')))

# Formating Input Data For Prediction.


if sel_itm_flag=='Yes':
    itm_flag=1
else:
    itm_flag=0
if sel_sxs=='Yes':
    sxs=1
else:
    sxs=0
if sel_cmp=='Yes':
    cmp=1
else:
    cmp=0
if sel_queue=='Large':
    queue=1
else:
    queue=0
if sel_peak_prod=='Yes':
    peak_prod=1
else:
    peak_prod=0

if sel_periodicity == 'CALENDAR':
    THI_PER_PERIODICITY = 1
else:
    THI_PER_PERIODICITY = 0


      
data = {'FME_NB_FACT': sel_facts,
            'FME_NB_MKT_HIERARCHY': sel_mkt_hie,
            'FME_NB_MBD': sel_mkt,
            'FME_NB_PRD_HIERARCHY': sel_prd_hie,
            'FME_NB_NODE': sel_prd_node,
            'FME_NB_ITEM': sel_items,
            'FME_NB_WEEK': sel_week,
            'FME_ITEM_FLAG': itm_flag,
            'FME_NB_FACT_PROMO': sel_fct_prd,
            'CHANNEL_COUNT': sel_cch,
            'FDS_SXS_ACTIVE': sxs,
            'MARKET_IS_COMPOSITE': cmp,
            'PEAK_PROD': peak_prod,
            'QUEUE_NAME': queue,
            'THI_PER_PERIODICITY':THI_PER_PERIODICITY
            }

inp = pd.DataFrame(data, index=[0])

# Slider
with st.form('Form1'):
    col2.header('Model Performance ')
    col3.header(' ')
    col3.header(' ')
    
# Data Preprocessing

df_cleaned = pd.read_csv(r'C:\Users\jayac\Downloads\Dataset\CIP\df_cleaned.csv')
df_cleaned.rename(columns={'SUM(D.DUR_MINS)':'DUR_MINS','MAX(D.PEAK_PROD)':'PEAK_PROD','MIN(D.QUEUE_NAME)':'QUEUE_NAME'},inplace=True)
df_cleaned.loc[df_cleaned['QUEUE_NAME'].isin(['PHY_AGG_L0','PHY_AGG_L1']),'QUEUE_NAME']='Large_Queue'
df_cleaned.loc[~df_cleaned['QUEUE_NAME'].isin(['Large_Queue']),'QUEUE_NAME']='Small_queue'
df_cleaned.loc[~df_cleaned['THI_PER_PERIODICITY'].isin(['WK']),'THI_PER_PERIODICITY']='Calendar_Periodicity'
df_cleaned.loc[df_cleaned['THI_PER_PERIODICITY'].isin(['WK']),'THI_PER_PERIODICITY']='Weekly_Periodicity'
df_cleaned.loc[df_cleaned['FME_ITEM_FLAG'].isin([2,3]),'FME_ITEM_FLAG']=1
df_cleaned['QUEUE_NAME']=np.where(df_cleaned['QUEUE_NAME']=='Large_Queue',1,0)
df_cleaned['FME_ITEM_FLAG']=df_cleaned['FME_ITEM_FLAG'].apply(lambda x: int(x))
df_cleaned['CHANNEL_COUNT']=df_cleaned['CHANNEL_COUNT'].apply(lambda x: int(x))
df_cleaned['THI_PER_PERIODICITY']=np.where(df_cleaned['THI_PER_PERIODICITY']=='Calendar_Periodicity',1,0)

train,test=train_test_split(df_cleaned,shuffle=True,test_size=0.25,random_state=123)
X_train=train.drop(columns=['MON_FDS_ID', 'FPS_FDS_RUN_ID', 'MON_TPR_ID', 'DUR_MINS',
       'FME_COU_CODE'])
y_train=train['DUR_MINS']
X_test=test.drop(columns=['MON_FDS_ID', 'FPS_FDS_RUN_ID', 'MON_TPR_ID', 'DUR_MINS', 
       'FME_COU_CODE'])
y_test=test['DUR_MINS']



lm = LinearRegression()
lm.fit(X_train, y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)
pred_test = abs(pred_test)
lmr2scoretrain = r2_score(y_train, pred_train)
lmr2scoretest = r2_score(y_test, pred_test)
st.write('### Algorithm Performance Comparision')
col4, col5 = st.beta_columns((1,1))
col4.subheader('Linear Reg  R2 Score(Train) ')
col4.write(lmr2scoretrain)
col5.subheader('Linear Reg  R2 Score(Test) ')
col5.write(lmr2scoretest)

dtr = DecisionTreeRegressor(min_samples_leaf= 3,
                                         random_state = 100)
dtr.fit(X_train, y_train)
dtrpred_train = dtr.predict(X_train)
dtrpred_test = dtr.predict(X_test)
dtrpred_test = abs(dtrpred_test)
dtrr2scoretrain = r2_score(y_train, dtrpred_train)
dtrr2scoretest = r2_score(y_test, dtrpred_test)
col4.subheader('Decision Tree R2 Score(Train) ')
col4.write(dtrr2scoretrain)
col5.subheader('Decision Tree R2 Score(Test) ')
col5.write(dtrr2scoretest)



rf= RandomForestRegressor( max_features=0.7, min_samples_leaf=3, min_samples_split=5,
                      n_estimators=60, n_jobs=-1,bootstrap=True,oob_score=True,criterion='mse')
rf.fit(X_train, y_train)
rfpred_train = rf.predict(X_train)
rfpred_test = rf.predict(X_test)
rfpred_test = abs(rfpred_test)
rfr2scoretrain = r2_score(y_train, rfpred_train)
rfr2scoretest = r2_score(y_test, rfpred_test)
col4.subheader('Random Forest R2 Score(Train) ')
col4.write(rfr2scoretrain)
col5.subheader('Random Forest R2 Score(Test) ')
col5.write(rfr2scoretest)


xg = XGBRegressor(gamma=0.05, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.05, max_delta_step=0, max_depth=6,booster='gbtree',n_estimators=80)
xg.fit(X_train, y_train)
xgpred_train = xg.predict(X_train)
xgpred_test = xg.predict(X_test)
xgpred_test = abs(xgpred_test)
xgr2scoretrain = r2_score(y_train, xgpred_train)
xgr2scoretest = r2_score(y_test, xgpred_test)
col4.subheader('XGBoost R2 Score(Train) ')
col4.write(xgr2scoretrain)
col5.subheader('XGBoost R2 Score(Test) ')
col5.write(xgr2scoretest)

tr = LGBMRegressor(num_leaves=25,
    max_depth=10,
    learning_rate=0.08,
    n_estimators=100)
tr.fit(X_train, y_train)
trpred_train = tr.predict(X_train)
trpred_test = tr.predict(X_test)
trpred_test = abs(trpred_test)
trr2scoretrain = r2_score(y_train, trpred_train)
trr2scoretest = r2_score(y_test, trpred_test)
col4.subheader('LGBM R2 Score(Train) ')
col4.write(trr2scoretrain)
col5.subheader('LGBM R2 Score(Test) ')
col5.write(trr2scoretest)


ct= CatBoostRegressor()
ct.fit(X_train, y_train)
ctrpred_train = ct.predict(X_train)
ctrpred_test = tr.predict(X_test)
ctrpred_test = abs(ctrpred_test)
ctrr2scoretrain = r2_score(y_train, ctrpred_train)
ctrr2scoretest = r2_score(y_test, ctrpred_test)
col4.subheader('Cat Boost R2 Score(Train) ')
col4.write(ctrr2scoretrain)
col5.subheader('Cat Boost R2 Score(Test) ')
col5.write(ctrr2scoretest)


vt = VotingRegressor([('rf',rf), ('tr',tr), ('xg',xg) ])
vt.fit(X_train, y_train)
vtpred_train = vt.predict(X_train)
vtpred_test = vt.predict(X_test)
vtpred_test = abs(vtpred_test)
vtr2scoretrain = r2_score(y_train, vtpred_train)
vtr2scoretest = r2_score(y_test, vtpred_test)
col4.subheader('Voting Regressor R2 Score(Train) ')
col4.write(vtr2scoretrain)
col5.subheader('Voting Regressor R2 Score(Test) ')
col5.write(vtr2scoretest)

st = StackingRegressor([('rf',rf), ('tr',tr), ('xg',xg)] , final_estimator=LinearRegression())
st.fit(X_train, y_train)
stpred_train = st.predict(X_train)
stpred_test = st.predict(X_test)
stpred_test = abs(stpred_test)
str2scoretrain = r2_score(y_train, stpred_train)
str2scoretest = r2_score(y_test, stpred_test)
col4.subheader('Stacking R2 Score(Train) ')
col4.write(str2scoretrain)
col5.subheader('Stacking R2 Score(Test) ')
col5.write(str2scoretest)




if sel_alg ==  'RandomForest':
    predrun = rf.predict(inp) 
elif sel_alg ==  'XGBoost':
    predrun = xg.predict(inp) 
elif sel_alg ==  'LightGBM':
    predrun = tr.predict(inp) 
elif sel_alg ==  'Voting Regressor':
    predrun = vt.predict(inp) 
else:  
    predrun = st.predict(inp) 

cost = round(predrun[0], 2) 

col1.subheader('RUNTIME PREDICTION')
col1.markdown('<font color="blue">As per given Data & Preference, Approx run time is</font>', unsafe_allow_html=True)
col1.write (cost)








