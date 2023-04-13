import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import MultiOutputRegressor

import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import base64
import pickle
import os.path
import math

from itertools import combinations
from joblib import Parallel, delayed

import os
from io import BytesIO

plt.rcParams['font.family'] = 'Malgun Gothic'

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Model building
def paste_seq(txt):
    data,line,y,sentence = {},[],-1,""
    
    for i in txt:
        if i == '\t':
            line.append(sentence)
            sentence = ""
        elif i == '\n':
            if sentence != "": line.append(sentence)
            data[y] = line
            line,y,sentence = [],y+1,""
        else:
            sentence += i
    
    df = pd.DataFrame.from_dict(data=data, orient='index')
    df.columns = df.loc[-1]
    df = df.drop(-1)
    df = df.replace("",np.nan)
    df = df.apply(pd.to_numeric, errors='ignore')
    
    return df

def rfe_seq(df3,Selected_X,Selected_y,ml_library):
    x = df3[Selected_X]
    y = df3[Selected_y]

    X_train_, X_test_, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
    
    # CTP Importance
    
    result_rank = pd.DataFrame(columns=["Score"])
    
    for col in Selected_X:
        X_train = X_train_[[col]]
        X_test = X_test_[[col]]
        
        scaler = StandardScaler().fit(X_train)
        X_train_r = scaler.transform(X_train)
        X_test_r = scaler.transform(X_test)
        
        model = ml_library.loc['RandomForest','model']
        if len(y) > 1: model = MultiOutputRegressor(model)
        
        model.fit(X_train_r, y_train)
        
        predictions = model.predict(X_train_r)
        predictions2 = model.predict(X_test_r)
        
        r_squared = sm.r2_score(y_train,predictions)
        if r_squared < 0: r_squared = 0
        r_squared2 = sm.r2_score(y_test,predictions2)
        if r_squared2 < 0: r_squared2 = 0
        
        result_rank.loc[col,"Score"] = (r_squared+r_squared2)/2
    
    result_rank = result_rank.sort_values("Score", ascending=False)
    
    # 상위 10개 항목을 대상으로 Combination 생성
    Selected_Xr = result_rank.index[:10].tolist()
    
    X_train_ = X_train_[Selected_Xr]
    X_test_ = X_test_[Selected_Xr]
    
    # CTP Combination test
    
    test_list = []
    
    for col_ea in range(X_train_.shape[1]):
        for cl in list(combinations(X_train_.columns, col_ea+1)):
            test_list.append(list(cl))
    
    def r2_seq(cl):
        X_train = X_train_[cl]
        X_test = X_test_[cl]
        
        scaler = StandardScaler().fit(X_train)
        X_train_r = scaler.transform(X_train)
        X_test_r = scaler.transform(X_test)
        
        model = ml_library.loc['RandomForest','model']
        if len(y) > 1: model = MultiOutputRegressor(model)
        
        model.fit(X_train_r, y_train)
        
        predictions = model.predict(X_train_r)
        predictions2 = model.predict(X_test_r)
        
        r_squared = sm.r2_score(y_train,predictions)
        if r_squared < 0: r_squared = 0
        r_squared2 = sm.r2_score(y_test,predictions2)
        if r_squared2 < 0: r_squared2 = 0
        
        return cl, (r_squared+r_squared2)/2, r_squared2
    
    with Parallel(n_jobs=12) as parallel:
        results = parallel(delayed(r2_seq)(i) for i in test_list)
    
    result = np.reshape([j for i in results for j in i],(-1,3))
    
    result_df = pd.DataFrame(result, columns=["CTP_list","R2_total","R2_test"])
    result_df = result_df.sort_values("R2_total", ascending=False).reset_index(drop=True)
    result_df.index = range(1,len(result_df)+1,1)
    
    return result_rank.sort_values("Score"),result_df

def build_model(df3,Selected_ml,Selected_X,Selected_y,ml_library):
    
    X = df3[Selected_X] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    
    models = []
    
    for mli in Selected_ml:
        if len(Selected_y) == 1: models.append((mli, Pipeline([('Scaler', StandardScaler()),(mli,ml_library.loc[mli,'model'])])))
        else: models.append((mli, MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),(mli,ml_library.loc[mli,'model'])]))))
    
    results,names,msg,mean,max1,min1,std = [],[],[],[],[],[],[]
    
    for name, model in models:
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
        
        for i, element in enumerate(cv_results):
            if element <= 0.0:
                cv_results[i] = 0.0
        
        results.append(abs(cv_results))
        
        names.append(name)
        msg.append('%s' % (name))
        mean.append('%f' %  (cv_results.mean()))
        min1.append('%f' %  (cv_results.min()))
        max1.append('%f' %  (cv_results.max()))
        std.append('%f' % (cv_results.std()))
        
    F_result = pd.DataFrame(np.transpose(msg))
    F_result.columns = ['Machine_Learning_Model']
    F_result['R2_Mean'] = pd.DataFrame(np.transpose(mean))
    F_result['R2_Std'] = pd.DataFrame(np.transpose(std))
    F_result['R2_Min'] = pd.DataFrame(np.transpose(min1))
    F_result['R2_Max'] = pd.DataFrame(np.transpose(max1))
    
    F_result[['R2_Mean','R2_Std','R2_Min','R2_Max']] = F_result[['R2_Mean','R2_Std','R2_Min','R2_Max']].astype('float')
    
    F_result = F_result.fillna(0)
    
    F_result = F_result.sort_values(by='R2_Mean', ascending=False)
    F_result = F_result.reset_index(drop = True)
    
    return F_result

def Opti_model(Model,df3,param_grid,Selected_X2,Selected_y,ml_library):
    
    X = df3[Selected_X2]
    Y = df3[Selected_y]

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)
    
    model = ml_library.loc[Model,'model']
    
    if len(Selected_y) == 1: grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    else: grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
    
    grid.fit(rescaled, Y)
    
    return grid.best_params_#grid.best_params_['n_estimators'],grid.best_params_['max_features']

def download_model(k, model):
    if k==0:
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/output_model;base64,{b64}" download="02_Trained_model.pkl">Download Trained Model.pkl</a>'
        st.markdown(href, unsafe_allow_html=True)
    elif k==1:
        model.save('test.h5')
        st.write('Currently, Neural network model save is underway.')
        st.write('If you want to make Neural network model, please contact Simulation team.')

#Excel로 변환 후 다운로드 함수, csv는 sheet 한장밖에 사용 못함. 고로 excel로 변환 적용.
def download_data_xlsx(df1, df2):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df1.to_excel(writer, sheet_name="CTP_list", index = False)
    df2.to_excel(writer, sheet_name="CTQ_list", index = False)
    writer.save()
    b64 = base64.b64encode(output.getvalue())
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="01_Data_for_Training.xlsx">Download Training Data.xlsx</a>', unsafe_allow_html=True)
    


#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################

def app(session_in):
    # 머신러닝 라이브러리 : 신규 라이브러리 추가 시 아래 DataFrame에 column 양식에 맞게 등록
    ml_library = pd.DataFrame(columns=['model','params','p1','p1_s','p2','p2_s'])
    
    ml_library.loc['Linear Regression'] = [LinearRegression(),None,None,None,None,None]
    ml_library.loc['Lasso'] = [Lasso(),None,None,None,None,None]
    ml_library.loc['Decision_Tree'] = [DecisionTreeRegressor(random_state=7),None,None,None,None,None]
    ml_library.loc['KNN'] = [KNeighborsRegressor(),
                             ['n_neighbors'],
                             ['Number of neighbers',2,10,2,8,2],
                             ['Step size for n_neighbors',1],None,None]
    ml_library.loc['GBM'] = [GradientBoostingRegressor(n_estimators=75,random_state=7),
                             ['n_estimators','max_features'],
                             ['Number of estimators (n_estimators)',1,301,11,151,20],
                             ['Step size for n_estimators',20],
                             ['Max features (max_features)',1,1,3,1],
                             ['Step size for max_features',1]]
    ml_library.loc['Extra Trees'] = [ExtraTreesRegressor(random_state=7),
                                     ['n_estimators','max_features'],
                                     ['Number of estimators (n_estimators)',1,301,11,151,20],
                                     ['Step size for n_estimators',20],
                                     ['Max features (max_features)',1,1,3,1],
                                     ['Step size for max_features',1]]
    ml_library.loc['RandomForest'] = [RandomForestRegressor(random_state=7),
                                            ['n_estimators','max_features'],
                                            ['Number of estimators (n_estimators)',1,301,11,151,20],
                                            ['Step size for n_estimators',20],
                                            ['Max features (max_features)',1,1,3,1],
                                            ['Step size for max_features',1]]
    ml_library.loc['XGBOOST'] = [xgboost.XGBRegressor(booster='gbtree',n_estimators= 100,random_state=7),
                                 ['n_estimators','max_depth'],
                                 ['Number of estimators (n_estimators)',1,301,41,101,20],
                                 ['Step size for n_estimators',20],
                                 ['max_depth',0,10,2,5,1],
                                 ['Step size for max_depth',1]]
    ml_library.loc['AB'] = [AdaBoostRegressor(random_state=7),
                            ['n_estimators','learning_rate'],
                            ['Number of estimators (n_estimators)',1,301,11,151,20],
                            ['Step size for n_estimators',20],
                            ['learning_rate',0.1,2.0,0.1,0.6,0.2],
                            ['Step size for learing_rate',0.2]]
    ml_library.loc['CatBoost'] = [CatBoostRegressor(random_seed=7,random_strength=7),
                                  ['learning_rate','depth'],
                                  ['learning_rate',0.1,2.0,0.5,1.5,0.2],
                                  ['Step size for learing_rate',0.2],
                                  ['depth',0,10,4,7,1],
                                  ['Step size for depth',1]]
    ml_library.loc['LGBM'] = [LGBMRegressor(seed=7),
                              ['learning_rate','num_iterations'],
                              ['learning_rate',0.1,2.0,0.5,1.5,0.2],
                              ['Step size for learing_rate',0.2],
                              ['num_iterations',50,500,100,300,1],
                              ['Step size for num_iterations',50]]
    
    with st.expander("Stage2. 머신러닝 모델 생성하기 - 가이드"):
        st.markdown("**1. 학습 데이터**")
        st.markdown("- Stage1 단계가 정상적으로 완료되었을 경우, 데이터 자동으로 로딩됩니다.")
        st.markdown("- Stage1 실행하지 않고 가공 완료된 데이터를 직접 붙여넣기도 가능합니다.")
        st.write("")
        st.markdown("**2. 공정인자(CTP), 품질인자(CTQ) 선정**")
        st.markdown("- CTP, CTQ 인자를 선정하고, 시각화 기능을 통해 인자 간 관계를 확인합니다.")
        st.write("")
        st.markdown("**3. 핵심 공정인자(CTP) 선정**")
        st.markdown("- RFE Method 분석 결과를 통해 CTP 별 CTQ에 끼치는 영향력을 확인합니다.")
        st.markdown("- 단일 CTQ인 경우, CTP 최적 개수를 추천해주며, CTQ 영향력을 기준으로 CTP를 자동으로 선정해줍니다.")
        st.markdown("- 다중 CTQ인 경우, CTP 최적 개수를 수동으로 입력 시 CTQ 영향력을 기준으로 CTP를 자동으로 선정해줍니다.")
        st.markdown("- CTP 수동으로 추가 및 제거가 가능하며, CTQ는 2단계에서 수정해야합니다.")
        st.write("")
        st.markdown("**4. 머신러닝 모델 비교**")
        st.markdown("- 각 ML 모델별 $R^2$ 정확도 및 편차 결과를 나타냅니다.")
        st.write("")
        st.markdown("**5. 모델 최적화**")
        st.markdown("- 모델의 정확도를 향상시키기 위해 4단계에서 분석된 모델을 최적화합니다.")
        st.markdown("- 단일 CTQ의 경우, Voting 기법과 하이퍼파라미터 최적화 기법을 사용할 수 있으며, 더 나은 결과를 보이는 기법을 선정하면 됩니다.")
        st.markdown("- 다중 CTQ의 경우, 하이퍼파라미터 최적화 기법을 사용하여 모델을 최적화할 수 있습니다.")
        st.markdown("- 모델 최적화 완료 후, 최종 모델 확정 버튼을 Click 합니다.")
        st.markdown("- CTP, CTQ 데이터 및 모델은 자동 저장되며, 다운로드 링크를 활용하여 별도 저장할 수 있습니다.")
        st.markdown("- Go to Stage3 버튼을 Click하여 다음 Stage로 이동합니다.")
    
    st.write("")
    #st.markdown("<h3 style='text-align: left; color: black;'>Stage2. 머신러닝모델 생성하기</h3>", unsafe_allow_html=True)
    tic1,tic2,tic3 = st.columns(3)
    with tic1: st.image('./pictures/stage1_m.png',use_column_width='always')
    with tic2: st.image('./pictures/stage2_c.png',use_column_width='always')
    with tic3: st.image('./pictures/stage3_m.png',use_column_width='always')
    st.write("")
    st.write("")

    #사용자가 접속한 시간'session_in'을 session_state로 저장한 후 본 앱에 사용할 변수(d11~d14) 선언
        
    list1 = 'list' + session_in + '_1'
    if list1 not in st.session_state: st.session_state[list1] = pd.DataFrame()
    
    list2 = 'list' + session_in + '_2'
    if list2 not in st.session_state: st.session_state[list2] = pd.DataFrame()
    
    output_data = 'output' + session_in + '_1'
    if output_data not in st.session_state: st.session_state[output_data] = pd.DataFrame()
    
    output_data2 = 'output' + session_in + '_2'
    if output_data2 not in st.session_state: st.session_state[output_data2] = [pd.DataFrame(),None,None]
    
    df = st.session_state[output_data]
    
    if len(df) == 0:
        st.error("데이터가 없습니다")
        st.markdown("**_Option.1 : Stage1. 을 먼저 진행하세요._**")
        st.markdown("**_※ [F5]키 누르면 앞 단계에서 저장한 데이터가 모두 초기화됩니다._**")
        st.write("")
        st.markdown("**_Option.2 : 데이터 수동 업로드_**")
        
        txt = st.text_area("엑셀 데이터영역 붙여넣기")
        
        if st.button("엑셀 데이터영역 붙여넣은 후 Click"):
            paste_res = paste_seq(txt)
            if paste_res is not None: st.session_state[output_data] = paste_res
            st.experimental_rerun()
        
    else:
        st.subheader('**1. 학습 데이터**')
        st._legacy_dataframe(df)
        
#=========================================================================================================================================================================
        st.write("")
        st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)
        st.write("")
        
        st.subheader('**2. 공정인자(CTP), 품질인자(CTQ) 선정**')
        st.caption("**CTP : X 인자  /  CTQ : Y 인자**")
        st.write("")
        
        df2 = df.copy()
        
        x = list(df2.columns)
        
        lv1c1,lv1c2 = st.columns(2)
        
        with lv1c1:
            st.markdown("**2.1 공정인자(CTP) 선택**")
            Selected_X = st.multiselect("※ 제거 시 우측의 품질인자(CTQ) 항목으로 이동", x, x)
            y = [a for a in x if a not in Selected_X]
        
        with lv1c2:
            st.markdown("**2.2 품질인자(CTQ) 선택**")
            Selected_y = st.multiselect("선택 및 제거", y, y)
            Selected_X = np.array(Selected_X)
            Selected_y = np.array(Selected_y)
        
        df22 = pd.concat([df2[Selected_X],df2[Selected_y]], axis=1) #########################
        
        st.write("")
        
        st.markdown('**※ CTP, CTQ 시각화**')
        
        st.write("▼ 상관관계 확인하기")
        if st.button('상관관계 분석'):
            corr = df22.corr()
            
            see1, see2 = st.columns(2)
            
            with see1:
                plt.figure()
                
                sns.set(rc = {'figure.figsize':(6,6)}, font="Malgun Gothic", font_scale=0.5)
                sns.heatmap(corr,  vmax=1, square=True, cbar_kws={"shrink": 0.7}, annot = True, cmap='coolwarm', linewidths=.5)
                
                st.pyplot(plt)
            
            with see2:
                st._legacy_dataframe(corr)
            
        st.write("")
        st.write("")
        st.write("▼ 데이터 시각화")
        
        visual = st.multiselect('인자 선택 및 제거',['모든 인자 보기'] + list(df22.columns))
        if visual == ['모든 인자 보기']: visual = df22.columns
        
        if len(visual) > 0:
            if st.button('데이터 시각화하기'):
                ino = math.ceil(len(visual) / 2)
                cno = 4
                
                fig, axs = plt.subplots(ino, cno, figsize=(10,3*ino))
                
                inoc = 0
                cnoc = 0
                
                for vis in visual:
                    if ino == 1: axl = axs[cnoc]
                    else: axl = axs[inoc,cnoc]
                    sns.distplot(df22[vis], hist_kws={'alpha':0.5}, bins=8, kde_kws={'color':'xkcd:purple','lw':3}, ax=axl)
                    axl.set_title('X variable distribution', fontsize=8)
                    cnoc += 1
                    
                    if ino == 1: axl = axs[cnoc]
                    else: axl = axs[inoc,cnoc]
                    sns.scatterplot(x=df22[vis],y=df22.iloc[:,-1],s=60,color='red', ax=axl)
                    sns.regplot(x=df22[vis],y=df22.iloc[:,-1], scatter=False, ax=axl)
                    axl.set_title('X - Y Graph', fontsize=8)
                    cnoc += 1
                    
                    if cnoc == cno:
                        inoc += 1
                        cnoc = 0
                
                if len(visual) %2 == 1:
                    try:
                        axs[2].axis('off')
                        axs[3].axis('off')
                    except:
                        axs[ino-1,2].axis('off')
                        axs[ino-1,3].axis('off')
                
                fig.tight_layout()
                st.pyplot(fig)
                        
#=========================================================================================================================================================================
        st.write("")
        st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)
        st.write("")
        
        st.subheader('3. 핵심 공정인자(CTP) 선정')
        st.write("")
        
        rfe_ss = 'rfe' + session_in + '_0'
        if rfe_ss not in st.session_state: st.session_state[rfe_ss] = None
        
        if st.button('CTP 분석 시작'):
            st.session_state[rfe_ss] = rfe_seq(df22,Selected_X,Selected_y,ml_library)
        
        st.write("")
        if st.session_state[rfe_ss] is not None:
            myc0,myc1,myc2 = st.columns(3)
            
            with myc0:
                st.markdown("**3.1 CTP 별 영향도 분석**")
                st.write("")
                
                sns.set(font="Malgun Gothic")
                st.session_state[rfe_ss][0].plot(kind='barh',figsize=(3,2.5)).legend(fontsize=6,loc='lower center')
                plt.xticks(size=6)
                plt.yticks(size=6)
                plt.tight_layout()
                st.pyplot(plt)
            
            with myc1:
                st.markdown("**3.2 CTP 조합 별 예상 정확도($R^2$)**")
                st.write("")
                st.write("")
                st.dataframe(st.session_state[rfe_ss][1])
                
            with myc2:
                st.markdown("**3.3 최종 CTP & CTQ 확인**")
                st.write("")
                NumXr = st.selectbox("▼ 최적 CTP list 순번 선택",st.session_state[rfe_ss][1].index.tolist(),0)
                feat_importances = st.session_state[rfe_ss][1].loc[NumXr,"CTP_list"]
                
                st.write("")
                
                ctn1 = st.container()
                
                Selected_X2 = st.multiselect("선택 및 제거", list(Selected_X), feat_importances)
                with ctn1:
                    fi = st.session_state[rfe_ss][1].copy()
                    fi["CTP_list"] = fi["CTP_list"].apply(set)
                    fi = fi[fi["CTP_list"]==set(Selected_X2)]
                    if len(fi) > 0: ino,perc = str(fi.index.values),str(round(fi.iloc[0]["R2_total"],3))
                    else: ino,perc = "","결과없음"
                    
                    st.markdown("**_▼최종 공정인자(CTP) 선정 > 조합 순번 : "+ino+", 예상 정확도($R^2$) : "+perc+"_**")
                    
                st.write("")
                
                st.markdown('**_▼최종 품질인자(CTQ) 개수 : %d_**' %len(Selected_y))
                st.multiselect("※ 선택 및 제거는 2.2 단계에서 수행", list(Selected_y), list(Selected_y))
                
                df3 = pd.concat([df22[Selected_X2],df22[Selected_y]], axis=1) #df22 -> df3 생성
        
#=========================================================================================================================================================================
        st.write("")
        st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)
        st.write("")

        st.subheader('4. 머신러닝 모델 비교')
        st.write("")
        
        st.markdown("**4.1. 알고리즘 선택하기**")
        
        ml = ml_library.index.tolist()
        
        Selected_ml = st.multiselect('선택 및 제거', ml, ml)
        
        model_compare = 'model_compare' + session_in + '_0'
        if model_compare not in st.session_state: st.session_state[model_compare] = None
        
        if st.button('모델 비교하기'):
            M_list = build_model(df3,Selected_ml,Selected_X2,Selected_y,ml_library)
            st.session_state[model_compare]  = M_list.copy()
            M_list_names = list(M_list['Machine_Learning_Model'])
            M_list = list(M_list['Machine_Learning_Model'][:3])
            st.session_state[list1] = M_list
            st.session_state[list2] = M_list_names
        
        if st.session_state[model_compare] is not None:
            F2_result = st.session_state[model_compare]
            st.write("")
            st.markdown("**4.2. 머신러닝 모델 비교 결과**")
            st.markdown("**모델 검증 방법 :  _K-Fold Cross Validation_**")
            st.markdown("**교차검증 샘플 분할 갯수 : _5, 검증 인자 : $R^2$_**")
            
            ccl1,ccl2 = st.columns([1,2])
            
            with ccl1:
                st.write("")
                st._legacy_dataframe(F2_result)
            
            with ccl2:
                fig, axs = plt.subplots(ncols=2, figsize=(9,5))
                g = sns.barplot(x="Machine_Learning_Model", y="R2_Mean", data=F2_result, ax=axs[0])
                g.set_xticklabels(g.get_xticklabels(), rotation=90)
                g.set(ylim=(0,1))
               
                g2 = sns.barplot(x="Machine_Learning_Model", y="R2_Std", data=F2_result, ax=axs[1])
                g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
                g2.set(ylim=(0,1))
                
                fig.tight_layout()
                st.pyplot(fig)
            
#=========================================================================================================================================================================
        if st.session_state[model_compare] is not None:
            st.write("")
            st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)
            st.write("")
            
            st.subheader('5. 모델 최적화')
            
            X = df3[Selected_X2]
            Y = df3[Selected_y]
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
            
            if Selected_y.shape[0] == 1:
                st.markdown('**※ Voting 기법과 하이퍼파라미터 최적화 중 택1**')
                st.write("")
                
                mfcol1,mfcol2 = st.columns(2)
                selm1,selm2 = st.columns(2)
                selp0,selp_gap1,selp1,selp_gap2,selp2 = st.columns([15,1,7,1,7])
                btnc1,btnc2 = st.columns(2)
                resc1,resc2 = st.columns(2)
                
                with mfcol1:
                    st.markdown('**5.1. 모델 최적화(Voting기법)**')
                
                with selm1:
                    ml2 = st.session_state[list1]
                    if len(ml2) == 0:
                        ml2 = ['Extra Trees','GBM','RandomForest']
                    
                    Selected_ml2 = st.multiselect('최적화 모델 선택 및 제거', ml2, ml2)
                    Selected_ml3 = list(Selected_ml2)
                    
                    with btnc1:
                        vtng = 'vtng' + session_in + '_0'
                        if vtng not in st.session_state: st.session_state[vtng] = None
                        
                        if st.button('최적화 모델 확인(Voting)'):
                            models =[]
                            
                            for mli in Selected_ml3:
                                models.append((mli, Pipeline([('Scaler', StandardScaler()),(mli,ml_library.loc[mli,'model'])])))
                                
                            model = VotingRegressor(estimators=models).fit(X,Y)
                            
                            results = []
                            
                            msg = []
                            mean = []
                            std = []        
                            
                            kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                            cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
                            
                            for i, element in enumerate(cv_results):
                                if element <= 0.0:
                                    cv_results[i] = 0.0
                                
                            results.append(abs(cv_results))
                            msg.append('%s' % model)
                            mean.append('%f' %  (cv_results.mean()))
                            std.append('%f' % (cv_results.std()))
                            
                            F_result3 = pd.DataFrame(np.transpose(msg))
                            F_result3.columns = ['Machine_Learning_Model']
                            F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                            F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                            
                            R2_mean = list(F_result3['R2_Mean'].values)
                            R2_std = list(F_result3['R2_Std'].values)
                            
                            st.session_state[vtng] = [R2_mean[0], R2_std[0], model]
                            
                            st.experimental_rerun()
                        
                        if st.session_state[vtng] is not None:
                            st.write("")
                            st.write("")
                            st.write("")
                            
                            R2_mean = st.session_state[vtng][0]
                            R2_std = st.session_state[vtng][1]
                            model = st.session_state[vtng][2]
                            
                            X = df3.iloc[:,:-1] # Using all column except for the last column as X
                            Y = df3.iloc[:,-1] # Selecting the last column as Y
                            
                            predictions = model.predict(X)
                            predictions = pd.DataFrame(predictions).values    
                            
                            st.markdown('**_모델 최적화 결과_**')
                            st.markdown('**· 정확도 및 편차 확인**')
                            
                            st.write('Voting 모델 정확도 ($R^2$):')
                            
                            st.info(R2_mean)
                            
                            st.write('모델 정확도 편차 (Standard Deviation):')
                            
                            st.info(R2_std)
                            
                            plt.figure(figsize=(10,10))
                            plt.plot(Y, Y, color='#0e4194', label = 'Actual data')
                            plt.scatter(Y, predictions, color='red', label = 'Prediction')
                            plt.xlabel(Selected_y[0])
                            plt.ylabel(Selected_y[0])
                            plt.legend()
                            st.pyplot(plt)
                            
                            if st.button("최종 모델 확정 - Voting 기법"):
                                st.session_state[output_data2] = [df3[Selected_X2],df3[Selected_y],st.session_state[vtng][2]]
                                
                                st.write("")
                                st.markdown('**_· Stage2 데이터 및 모델 자동 저장 완료_**')
                                st.write("")
                                
                                if st.button("▶Go to Stage3"):
                                    st.experimental_rerun()
                                
                                st.write("")
                                st.markdown('**_· 학습 데이터(.xlsx) & 학습 완료 모델(.pkl) 저장하기_**')
                                
                                download_data_xlsx(df3[Selected_X2],df3[Selected_y])
                                download_model(0,model)
                                st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                                
                with mfcol2:
                    st.markdown('**5.2. 하이퍼파라미터 최적화**')
                    
                with selm2:
                    M_list_names = ['KNN']
                    
                    if len(st.session_state[list2]) != 0: M_list_names = st.session_state[list2]
                    
                    max_num = df3.shape[1]
                    
                    Model = st.selectbox('하이퍼파라미터 튜닝 - 알고리즘 선택하기',M_list_names)
                    
                    if Model in ['Linear Regression','Lasso','Decision_Tree']:
                        st.markdown('_해당 알고리즘은 해당되지 않습니다._')
                        
                    elif Model == 'KNN':
                        param_list = ml_library.loc[Model,'params']
                        
                        with selp1:
                            p1 = st.slider(ml_library.loc[Model,'p1'][0],
                                           ml_library.loc[Model,'p1'][1],ml_library.loc[Model,'p1'][2],
                                           (ml_library.loc[Model,'p1'][3],ml_library.loc[Model,'p1'][4]),
                                           ml_library.loc[Model,'p1'][5])
                        
                        with selp2: p1_s = st.slider(ml_library.loc[Model,'p1_s'][0],ml_library.loc[Model,'p1_s'][1])
                        
                        p1_range = np.arange(p1[0], p1[1]+p1_s, p1_s)
                        param_grid = {param_list[0]:p1_range}
                    
                    else:
                        param_list = ml_library.loc[Model,'params']
                        
                        with selp1: p1 = st.slider(ml_library.loc[Model,'p1'][0],
                                                   ml_library.loc[Model,'p1'][1],ml_library.loc[Model,'p1'][2],
                                                   (ml_library.loc[Model,'p1'][3],ml_library.loc[Model,'p1'][4]),
                                                   ml_library.loc[Model,'p1'][5])
                        
                        with selp1: p1_s = st.slider(ml_library.loc[Model,'p1_s'][0],ml_library.loc[Model,'p1_s'][1])
                        
                        with selp2:
                            if Model in ['GBM','Extra Trees','RandomForest']:
                                p2 = st.slider(ml_library.loc[Model,'p2'][0],
                                               ml_library.loc[Model,'p2'][1],max_num,
                                               (ml_library.loc[Model,'p2'][2],ml_library.loc[Model,'p2'][3]),
                                               ml_library.loc[Model,'p2'][4])
                            else:
                                p2 = st.slider(ml_library.loc[Model,'p2'][0],
                                               ml_library.loc[Model,'p2'][1],ml_library.loc[Model,'p2'][2],
                                               (ml_library.loc[Model,'p2'][3],ml_library.loc[Model,'p2'][4]),
                                               ml_library.loc[Model,'p2'][5])
                        
                        with selp2: p2_s = st.slider(ml_library.loc[Model,'p2_s'][0],ml_library.loc[Model,'p2_s'][1])
                        
                        p1_range = np.arange(p1[0], p1[1]+p1_s, p1_s)
                        p2_range = np.arange(p2[0], p2[1]+p2_s, p2_s)
                        param_grid = {param_list[0]:p1_range, param_list[1]:p2_range}
                        
                    with btnc2:
                        hptn = 'hptn' + session_in + '_0'
                        if hptn not in st.session_state: st.session_state[hptn] = None
                        
                        if st.button('최적화 모델 확인(하이퍼파라미터 튜닝)'):
                            if Model in ['Linear Regression','Lasso','Decision_Tree']:
                                best_grid = "해당 없음"
                                model = Pipeline([('Scaler', StandardScaler()),(Model,ml_library.loc[Model,'model'])])
                            
                            elif Model == 'KNN':
                                best_grid = Opti_model(Model,df3,param_grid,Selected_X2,Selected_y,ml_library)
                                m_comb = {'KNN':KNeighborsRegressor(n_neighbors=best_grid[param_list[0]])}
                                model = Pipeline([('Scaler', StandardScaler()),(Model,m_comb[Model])])
                            
                            elif Model in ['GBM','Extra Trees','RandomForest','XGBOOST','AB','CatBoost','LGBM']:
                                best_grid = Opti_model(Model,df3,param_grid,Selected_X2,Selected_y,ml_library)
                                m_comb = {'GBM':GradientBoostingRegressor(n_estimators=best_grid[param_list[0]], max_features=best_grid[param_list[1]],random_state=7),
                                          'Extra Trees':ExtraTreesRegressor(n_estimators=best_grid[param_list[0]], max_features=best_grid[param_list[1]],random_state=7),
                                          'RandomForest':RandomForestRegressor(n_estimators=best_grid[param_list[0]], max_features=best_grid[param_list[1]],random_state=7),
                                          'XGBOOST':xgboost.XGBRegressor(booster='gbtree',n_estimators=best_grid[param_list[0]], max_depth=best_grid[param_list[1]],random_state=7),
                                          'AB':AdaBoostRegressor(n_estimators=best_grid[param_list[0]], learning_rate=best_grid[param_list[1]],random_state=7),
                                          'CatBoost':CatBoostRegressor(learning_rate=best_grid[param_list[0]], depth=best_grid[param_list[1]],random_seed=7,random_strength=7),
                                          'LGBM':LGBMRegressor(learning_rate=best_grid[param_list[0]], num_iterations=best_grid[param_list[1]],seed=7)}
                                model = Pipeline([('Scaler', StandardScaler()),(Model,m_comb[Model])])
                            
                            model.fit(X_train,y_train)
                            
                            results = []
            
                            msg = []
                            mean = []
                            std = []        
                            
                            kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                            cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
                        
                            for i, element in enumerate(cv_results):
                                if element <= 0.0:
                                    cv_results[i] = 0.0
                            
                            results.append(abs(cv_results))
                            msg.append('%s' % Model)
                            mean.append('%f' %  (cv_results.mean()))
                            std.append('%f' % (cv_results.std()))
                            
                            F_result3 = pd.DataFrame(np.transpose(msg))
                            F_result3.columns = ['Machine_Learning_Model']
                            F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                            F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                            
                            R2_mean = list(F_result3['R2_Mean'].values)
                            R2_std = list(F_result3['R2_Std'].values)
                            
                            st.session_state[hptn] = [R2_mean[0], R2_std[0], model, best_grid]
                            
                            st.experimental_rerun()
                            
                        if st.session_state[hptn] is not None:
                            st.write("")
                            st.write("")
                            st.write("")
                            st.markdown('**_모델 최적화 결과_**')
                            
                            R2_mean = st.session_state[hptn][0]
                            R2_std = st.session_state[hptn][1]
                            model = st.session_state[hptn][2]
                            best_grid = st.session_state[hptn][3]
                            
                            if st.session_state[hptn] != 0: st.markdown("**· 최적 파라미터 : %s**" %(best_grid))
                            
                            predictions = model.predict(X)
                            predictions = pd.DataFrame(predictions).values    
                            
                            st.write('튜닝 모델 정확도 ($R^2$):')
                            
                            st.info(R2_mean)
                            
                            st.write('모델 정확도 편차 (Standard Deviation):')
                            
                            st.info(R2_std)
                            
                            plt.figure(figsize=(10,10))
                            plt.plot(Y.values, Y.values, color='#0e4194', label = 'Actual data')
                            plt.scatter(Y, predictions, color='red', label = 'Prediction')
                            plt.xlabel(Selected_y[0])
                            plt.ylabel(Selected_y[0])
                            plt.legend()
                            st.pyplot(plt)
                            
                            if st.button("최종 모델 확정 - 하이퍼파라미터 최적화"):
                                st.session_state[output_data2] = [df3[Selected_X2],df3[Selected_y],st.session_state[hptn][2]]
                                
                                st.write("")
                                st.markdown('**_· Stage2 데이터 및 모델 자동 저장 완료_**')
                                st.write("")
                                
                                if st.button("▶Go to Stage3"):
                                    st.experimental_rerun()
                                
                                st.write("")
                                st.markdown('**_· 학습 데이터(.xlsx) & 학습 완료 모델(.pkl) 저장하기_**')
                                
                                download_data_xlsx(df3[Selected_X2],df3[Selected_y])
                                download_model(0,model)
                                st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**") 
                                
            elif Selected_y.shape[0] > 1:
                st.markdown('**5.1. 하이퍼파라미터 최적화**')
                
                M_list_names = ['KNN']
                
                if len(st.session_state[list2]) != 0: M_list_names = st.session_state[list2]
                    
                max_num = df3.shape[1]
                
                Model = st.selectbox('하이퍼파라미터 튜닝 - 알고리즘 선택하기', M_list_names)
                
                selp1,selp2 = st.columns(2)
                
                if Model in ['Linear Regression','Lasso','Decision_Tree']:
                    st.markdown('_해당 알고리즘은 해당되지 않습니다._')
                
                elif Model == 'KNN':
                    param_list = ['estimator__' + i for i in ml_library.loc[Model,'params']]
                    
                    with selp1:
                        p1 = st.slider(ml_library.loc[Model,'p1'][0],
                                       ml_library.loc[Model,'p1'][1],ml_library.loc[Model,'p1'][2],
                                       (ml_library.loc[Model,'p1'][3],ml_library.loc[Model,'p1'][4]),
                                       ml_library.loc[Model,'p1'][5])
                    
                    with selp2: p1_s = st.slider(ml_library.loc[Model,'p1_s'][0],ml_library.loc[Model,'p1_s'][1])
                    
                    p1_range = np.arange(p1[0], p1[1]+p1_s, p1_s)
                    param_grid = {param_list[0]:p1_range}
                
                else:
                    param_list = ['estimator__' + i for i in ml_library.loc[Model,'params']]
                    
                    with selp1: p1 = st.slider(ml_library.loc[Model,'p1'][0],
                                               ml_library.loc[Model,'p1'][1],ml_library.loc[Model,'p1'][2],
                                               (ml_library.loc[Model,'p1'][3],ml_library.loc[Model,'p1'][4]),
                                               ml_library.loc[Model,'p1'][5])
                    
                    with selp1: p1_s = st.slider(ml_library.loc[Model,'p1_s'][0],ml_library.loc[Model,'p1_s'][1])
                    
                    with selp2:
                        if Model in ['GBM','Extra Trees','RandomForest']:
                            p2 = st.slider(ml_library.loc[Model,'p2'][0],
                                           ml_library.loc[Model,'p2'][1],max_num,
                                           (ml_library.loc[Model,'p2'][2],ml_library.loc[Model,'p2'][3]),
                                           ml_library.loc[Model,'p2'][4])
                        else:
                            p2 = st.slider(ml_library.loc[Model,'p2'][0],
                                           ml_library.loc[Model,'p2'][1],ml_library.loc[Model,'p2'][2],
                                           (ml_library.loc[Model,'p2'][3],ml_library.loc[Model,'p2'][4]),
                                           ml_library.loc[Model,'p2'][5])
                    
                    with selp2: p2_s = st.slider(ml_library.loc[Model,'p2_s'][0],ml_library.loc[Model,'p2_s'][1])
                    
                    p1_range = np.arange(p1[0], p1[1]+p1_s, p1_s)
                    p2_range = np.arange(p2[0], p2[1]+p2_s, p2_s)
                    param_grid = {param_list[0]:p1_range, param_list[1]:p2_range}
                
                hptn_m = 'hptn_m' + session_in + '_0'
                if hptn_m not in st.session_state: st.session_state[hptn_m] = None
                
                if st.button('최적화 모델 확인(하이퍼파라미터 튜닝)'):
                    if Model == 'Linear Regression' or Model == 'Lasso' or Model == 'Decision_Tree':
                        best_grid = "해당 없음"
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),(Model,ml_library.loc[Model,'model'])]))
                    
                    elif Model == 'KNN':
                        best_grid = Opti_model(Model,df3,param_grid,Selected_X2,Selected_y,ml_library)
                        m_comb = {'KNN':KNeighborsRegressor(n_neighbors=best_grid[param_list[0]])}
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),(Model,m_comb[Model])]))
                    
                    elif Model in ['GBM','Extra Trees','RandomForest','XGBOOST','AB','CatBoost','LGBM']:
                        best_grid = Opti_model(Model,df3,param_grid,Selected_X2,Selected_y,ml_library)
                        m_comb = {'GBM':GradientBoostingRegressor(n_estimators=best_grid[param_list[0]], max_features=best_grid[param_list[1]],random_state=7),
                                  'Extra Trees':ExtraTreesRegressor(n_estimators=best_grid[param_list[0]], max_features=best_grid[param_list[1]],random_state=7),
                                  'RandomForest':RandomForestRegressor(n_estimators=best_grid[param_list[0]], max_features=best_grid[param_list[1]],random_state=7),
                                  'XGBOOST':xgboost.XGBRegressor(booster='gbtree',n_estimators=best_grid[param_list[0]], max_depth=best_grid[param_list[1]],random_state=7),
                                  'AB':AdaBoostRegressor(n_estimators=best_grid[param_list[0]], learning_rate=best_grid[param_list[1]],random_state=7),
                                  'CatBoost':CatBoostRegressor(learning_rate=best_grid[param_list[0]], depth=best_grid[param_list[1]],random_seed=7,random_strength=7),
                                  'LGBM':LGBMRegressor(learning_rate=best_grid[param_list[0]], num_iterations=best_grid[param_list[1]],seed=7)}
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),(Model,m_comb[Model])]))
                    
                    model.fit(X_train,y_train)
                    
                    results = []
    
                    msg = []
                    mean = []
                    std = []
                    
                    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
                
                    for i, element in enumerate(cv_results):
                        if element <= 0.0:
                            cv_results[i] = 0.0
                    
                    results.append(cv_results)
                    msg.append('%s' % Model)
                    mean.append('%f' %  (cv_results.mean()))
                    std.append('%f' % (cv_results.std()))
                    
                    F_result3 = pd.DataFrame(np.transpose(msg))
                    F_result3.columns = ['Machine_Learning_Model']
                    F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                    F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                
                    R2_mean = list(F_result3['R2_Mean'].values)
                    R2_std = list(F_result3['R2_Std'].values)
                    
                    st.session_state[hptn_m] = [R2_mean[0], R2_std[0], model, best_grid]
                    
                    st.experimental_rerun()
                
                if st.session_state[hptn_m] is not None:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.markdown('**_모델 최적화 결과_**')
                    
                    R2_mean = st.session_state[hptn_m][0]
                    R2_std = st.session_state[hptn_m][1]
                    model = st.session_state[hptn_m][2]
                    best_grid = st.session_state[hptn_m][3]
                    
                    if st.session_state[hptn_m] != 0: st.markdown("**· 최적 파라미터 : %s**" %(best_grid))
                    
                    predictions = model.predict(X)
                    predictions = pd.DataFrame(predictions)
                    
                    mfresc1,mfresc2 = st.columns(2)
                    
                    with mfresc1:
                        st.write('튜닝  모델 정확도 ($R^2$):')
                        
                        st.info(R2_mean)
                    
                    with mfresc2:
                        st.write('모델 정확도 편차 (Standard Deviation):')
                        
                        st.info(R2_std)
                    
                    fig, axs = plt.subplots(ncols=Y.shape[1], figsize=(10,3))
                    
                    for i in range(1,Y.shape[1]+1):
                        plt.subplot(1,Y.shape[1],i)
                        
                        plt.plot(Y.iloc[:,i-1], Y.iloc[:,i-1], color='#0e4194', label = 'Actual data')
                        plt.scatter(Y.iloc[:,i-1], predictions.iloc[:,i-1], color='red', label = 'Prediction')
                        plt.title(Y.columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        
                    st.pyplot(fig)
                    
                    if st.button("최종 모델 확정 - 하이퍼파라미터 최적화"):
                        st.session_state[output_data2] = [df3[Selected_X2],df3[Selected_y],st.session_state[hptn_m][2]]
                        
                        st.write("")
                        st.markdown('**_· Stage2 데이터 및 모델 자동 저장 완료_**')
                        st.write("")
                        
                        if st.button("▶Go to Stage3"):
                            st.experimental_rerun()
                        
                        st.write("")
                        st.markdown('**_· 학습 데이터(.xlsx) & 학습 완료 모델(.pkl) 저장하기_**')
                        
                        #download_data_xlsx(df3[Selected_X2],df3[Selected_y])
                        download_model(0,model)
                        st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**") 
                        
