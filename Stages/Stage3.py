import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pickle
import base64
import random

import os

plt.rcParams['font.family'] = 'Malgun Gothic'

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    
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

def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv", itx="Download Dataset.csv"): 
    csv_exp = _df.to_csv(index=False, encoding='CP949')
    b64 = base64.b64encode(csv_exp.encode('CP949')).decode('CP949')  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" >' + itx + '</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def app(session_in):
    
    output_data2 = 'output' + session_in + '_2'
    
    with st.expander("Stage3. 최적 조건 예측하기 - 가이드"):
        st.markdown("**1. 업로드된 데이터 및 모델 확인**")
        st.markdown("- Stage2 단계가 정상적으로 완료되었을 경우, 데이터 및 모델이 자동으로 로딩됩니다.")
        st.markdown("- Stage2 실행하지 않고 기존에 다운로드 한 데이터 및 모델 파일을 직접 업로드할 수도 있습니다.")
        st.markdown("- 데이터 파일을 업로드하는 경우, 자동으로 CTP와 CTQ가 선정됩니다.")
        st.markdown("- 데이터 파일을 붙여넣는 경우, CTP와 CTQ를 수동으로 선정해야 합니다.")
        st.markdown("- 선정된 CTP, CTQ 인자 및 모델 정보를 확인합니다.")
        st.write("")
        st.markdown("**2. CTP, CTQ 예측**")
        st.markdown("- 최적 공정조건(CTP) 예측 : CTP 별 허용 범위를 선정, 해당 범위 내에서 목표 CTQ 값을 나타내는 CTP 값을 자동으로 계산")
        st.markdown("- 단일 품질결과(CTQ) 예측 : 단일 CTP 조건의 CTQ 값을 계산")
        st.markdown("- 다중 품질결과(CTQ) 예측 : CTP 조건 여러개를 입력, 각각의 CTQ 값을 계산")
    
    st.write("")
    #st.markdown("<h2 style='text-align: left; color: black;'>Stage3. 최적 조건 예측</h2>", unsafe_allow_html=True)
    tic1,tic2,tic3 = st.columns(3)
    with tic1: st.image('./pictures/stage1_m.png',use_column_width='always')
    with tic2: st.image('./pictures/stage2_m.png',use_column_width='always')
    with tic3: st.image('./pictures/stage3_c.png',use_column_width='always')
    st.write("")
    st.write("")
    
    if output_data2 not in st.session_state: st.session_state[output_data2] = [pd.DataFrame(),pd.DataFrame(),None]
    df_x = st.session_state[output_data2][0]
    model = st.session_state[output_data2][2]
    
    if len(df_x) == 0 or model is None:
        st.error("데이터 또는 모델이 없습니다")
        st.markdown("**_Option.1 : Stage1~2를 먼저 진행하세요._**")
        st.markdown("**_※ [F5]키 누르면 앞 단계에서 저장한 데이터가 모두 초기화됩니다._**")
        st.write("")
        st.markdown("**_Option.2 : 데이터 및 모델 수동 업로드_**")
        
        prec1,prec2,prec3 = st.columns(3, gap="large")
        
        with prec1:
            st.markdown("**데이터 업로드 방법1**")
            
            txt = st.text_area("엑셀 데이터영역 붙여넣기")
            if st.button("엑셀 데이터영역 붙여넣은 후 Click"):
                paste_res = paste_seq(txt)
                if paste_res is not None: st.session_state[output_data2][0] = paste_res
        
        with prec2:
            st.markdown("**데이터 업로드 방법2**")
            uploaded_file = st.file_uploader('Stage2에서 다운로드한 엑셀 파일', type=["xlsx"])
            if uploaded_file is not None:
                st.session_state[output_data2][0] = pd.read_excel(uploaded_file, sheet_name = 'CTP_list')
                st.session_state[output_data2][1] = pd.read_excel(uploaded_file, sheet_name = 'CTQ_list')
        
        with prec3:
            st.markdown("**모델 업로드**")
            uploaded_modelfile = st.file_uploader('Stage2에서 다운로드한 모델 파일', type=["pkl"])
            if uploaded_modelfile is not None:
                st.session_state[output_data2][2] = pickle.load(uploaded_modelfile)
        
        st.write("")
        
        pre2cg1,pre2c1,pre2cg2,pre2cg3,pre2c2,pre2cg4 = st.columns([2,2,2,1,1,1])
        
        with pre2c1:
            if len(st.session_state[output_data2][0]) == 0: st.markdown("**※ 데이터 업로드 필요**")
        
        with pre2c2:
            if st.session_state[output_data2][2] is None: st.markdown("**※ 모델 업로드 필요**")
        
        if len(st.session_state[output_data2][0]) != 0 and st.session_state[output_data2][2] is not None:
            if st.button("데이터 및 모델 업로드 완료"): st.experimental_rerun()
    else:
        df_x = st.session_state[output_data2][0]
        df_y = st.session_state[output_data2][1]
        df = pd.concat([df_x,df_y], axis=1)
        model = st.session_state[output_data2][2]
        
        st.subheader('**1. 업로드된 데이터 및 모델 확인**')
        
        st.markdown("**1.1 데이터 확인**")
        
        x = df_x.columns.tolist()
        if len(df_y) > 0: y = df_y.columns.tolist()
        else: y = []
        
        lv1c1,lv1c2 = st.columns(2)
        
        with lv1c1:
            st.markdown("**· 공정인자(CTP)**")
            Selected_X = st.multiselect("선택 및 제거", df.columns, x)
        
        with lv1c2:
            st.markdown("**· 품질인자(CTQ)**")
            Selected_y = st.multiselect("선택 및 제거", df.columns, y)
        
        datachk3 = 'datachk3' + session_in + '_0'
        if datachk3 not in st.session_state: st.session_state[datachk3] = False
        
        if len(df_y) == 0:
            if st.button("CTP & CTQ 확인 완료"):
                st.session_state[output_data2][0] = df[Selected_X]
                st.session_state[output_data2][1] = df[Selected_y]
                st.session_state[datachk3] = True
                st.experimental_rerun()
        else: st.session_state[datachk3] = True
        
        if st.session_state[datachk3] == True:
            st.write("")
            st.write("")
            st.markdown("**1.2 모델 확인**")
            
            model_review = 'model_review' + session_in + '_1'
            if model_review not in st.session_state: st.session_state[model_review] = [None,None,None]
            
            if st.session_state[model_review][0] is None:
                X_train = df[Selected_X]
                y_train = df[Selected_y]
                
                predictions = model.predict(X_train)
            
                results = []
        
                msg = []
                mean = []
                std = []        
                    
                kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
                
                for i, element in enumerate(cv_results):
                    if element <= 0.0:
                        cv_results[i] = 0.0
                        
                results.append(cv_results)
                
                msg.append('%s' % model)
                mean.append('%f' %  (cv_results.mean()))
                std.append('%f' % (cv_results.std()))
                        
                        
                F_result3 = pd.DataFrame(np.transpose(msg))
                F_result3.columns = ['Machine_Learning_Model']
                F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                
                R2_mean = list(F_result3['R2_Mean'].values)
                R2_std = list(F_result3['R2_Std'].values)
                
                
                model_txt = str(model)
                model_list = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','AB','XGBOOST','Extra Trees','RandomForest','CatBoost','LGBM']
                model_name = []
                for md in model_list:
                    if md in model_txt:
                        model_name.append(md)
                
                st.session_state[model_review] = [model_name,R2_mean[0],R2_std[0]]
                st.experimental_rerun()
            else:
                lv2c1,lv2c2,lv2c3 = st.columns(3, gap="large")
                
                with lv2c1:
                    st.write('모델명:')
                    st.info(st.session_state[model_review][0])
                
                with lv2c2:
                    st.write('모델 정확도 ($R^2$):')
                    st.info(st.session_state[model_review][1])
                
                with lv2c3:
                    st.write('모델 정확도 편차 (Standard Deviation):')
                    st.info(st.session_state[model_review][2])
            
    #=========================================================================================================================================================================        
            st.write("")
            st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)
            st.write("")
            
            st.subheader('2. CTP, CTQ 예측')
            
            pred_process = st.selectbox("예측 방법 선택",['Select ▼','최적 공정조건(CTP) 예측','단일 품질결과(CTQ) 예측','다중 품질결과(CTQ) 예측'])
            st.write("")
            st.write("")
            
            predc1,predc2 = st.columns(2, gap="large")
            
            if pred_process == '최적 공정조건(CTP) 예측':
                with predc1:
                    st.write('**▼ CTQ 목표 수치 입력**')
                    
                    ctql_v = []
                    ctql_n = []
                    
                    for column in Selected_y:
                        value = st.number_input(column, None, None, df[column].mean(),format="%.3f")
                        ctql_v.append(value)
                        ctql_n.append(column)
                    
                    F_result = pd.DataFrame({'Value':ctql_v}, index=ctql_n).transpose()
                    
                    st.write("")
                    st.write('**▼ CTP 범위 선정**')
                    
                    ctpl_v = {}
                    for column in Selected_X:
                        max1 = round(float(df[column].max()),3)
                        min1 = round(float(df[column].min()),3)
                        step = round((max1-min1)/20.0,3)
                        value = st.slider(column, min1, max1, (min1,max1), step)
                        ctpl_v[column] = value
                    
                    st.write("")
                    st.write('**▼ 생성할 샘플 개수 입력**')
                    
                    N_sample = st.number_input("샘플 개수",0, 10000, 1000, format="%d")
                    
                with predc2:
                    st.markdown("**▼ 최적 CTP 예측 결과**")
                    
                    st.write("")
                    st.write("")
                    
                    if st.button("최적 CTP 예측"):
                        st.write("")
                        
                        sample_l = []
                        
                        for column in Selected_X:
                            vmin = ctpl_v[column][0]
                            vmax = ctpl_v[column][1]
                            vdif = (vmax - vmin) / 100
                            sample_l.append(np.arange(vmin, vmax+vdif, vdif).tolist())
                        
                        sample_df = pd.DataFrame(columns=Selected_X)
                        
                        if len(Selected_X) == 1:
                            st.warning("생성할 샘플 개수가 경우의 수를 초과합니다.")
                            N_sample = 100
                        
                        seed = 1
                        while len(sample_df) < N_sample:
                            random.seed(seed)
                            sample_df.loc[len(sample_df)] = [random.choice(i) for i in sample_l]
                            sample_df = sample_df.drop_duplicates()
                            seed += 1
                        
                        sample_df = sample_df.reset_index(drop=True)
                        
                        sample_result = pd.DataFrame(model.predict(sample_df), columns=Selected_y)
                        
                        sample_df = pd.concat([sample_df,sample_result], axis=1)
                        
                        diff_c = 'diff(%)'
                        sample_df[diff_c] = 0
                        for column in Selected_y: sample_df[diff_c] = sample_df[diff_c] + abs(1 - sample_df[column] / F_result.loc[F_result.index,column].values)
                        
                        sample_df[diff_c] = round(sample_df[diff_c] * 100,2)
                        sample_df = sample_df.sort_values(diff_c).reset_index(drop=True)
                        
                        st.write('**· 최적 공정조건(CTP) :**')
                        
                        best_CTP = sample_df.loc[[0],Selected_X].copy()
                        best_CTP.index = ['Best_CTP']
                        st._legacy_dataframe(best_CTP)
                        
                        st.write('**· 전체 샘플 예측 결과 :**')
                        
                        sample_df2 = sample_df.copy()
                        multicolumnl = ['CTP' for i in Selected_X]+['CTQ' for i in Selected_y]+[diff_c]
                        sample_df2.columns = [multicolumnl,sample_df2.columns]
                        
                        st._legacy_dataframe(sample_df2)
                        
                        fig, axs = plt.subplots(len(Selected_y), 1, figsize=(5,3*len(Selected_y)))
                        
                        for i in range(len(Selected_y)):
                            column = Selected_y[i]
                            plt.subplot(len(Selected_y),1,i+1)
                            plt.scatter(sample_df.index, sample_df.loc[:,column], color='#0e4194')
                            plt.scatter(sample_df.index[0], sample_df.loc[0,column], color='#e30613')
                            lineval = F_result.loc[F_result.index,column].values
                            plt.plot([sample_df.index[0],sample_df.index[-1]], [lineval,lineval], color='red')
                            plt.title(column)
                        fig.tight_layout()
                        st.pyplot(fig)
            
            elif pred_process == '단일 품질결과(CTQ) 예측':
                pred2c1,pred2c2 = st.columns(2, gap="large")
                
                with predc1:
                    st.markdown("**▼ CTP 입력**")
                    
                    st.write("※ 각 CTP 별 예측하고자하는 값 입력")
                with pred2c1:
                    st.write("")
                    st.markdown("**· 입력 항목**")
                    
                    test = []
                    name = []
                    
                    for column in Selected_X:
                        value = st.number_input(column, None, None, df[column].mean(),format="%.3f")
                        test.append(value)
                        name.append(column)
                    
                    F_result = pd.DataFrame({'Value':test}, index=name).transpose()
                    
                with predc2:
                    st.markdown("**▼ 단일 CTQ 예측 결과**")
                    if st.button("단일 CTQ 예측"):
                        predictions = model.predict(F_result)
                        predictions = pd.DataFrame(predictions, columns = Selected_y)
                        predictions.index = ['Value']
                        
                        with pred2c2:
                            st.write("")
                            st.markdown("**· 입력한 CTP 확인**")
                            st.write("")
                            st.write("")
                            st.write(F_result)
                            
                            st.write("")
                            st.markdown("**· 단일 CTQ 예측결과**")
                            st.write(predictions)
            
            elif pred_process == '다중 품질결과(CTQ) 예측':
                pp3 = 'pp3' + session_in + '_1'
                if pp3 not in st.session_state: st.session_state[pp3] = [None,None]
                
                with predc1:
                    st.markdown("**▼ CTP 데이터 리스트 업로드**")
                    
                    txt = st.text_area("엑셀 데이터영역 붙여넣기")
                
                st.write("")
                
                pred2c1,pred2c2 = st.columns(2, gap="large")
                
                with pred2c1:
                    if st.button("엑셀 데이터영역 붙여넣은 후 Click"):
                        paste_res = paste_seq(txt)
                        if paste_res is not None: st.session_state[pp3] = [paste_res[Selected_X],None]
                        st.experimental_rerun()
                    
                    st.write("")
                    if st.session_state[pp3][0] is not None:
                        st.markdown("**▼ 업로드 된 CTP 데이터 리스트**")
                        ctp_df = st.session_state[pp3][0]
                        st._legacy_dataframe(ctp_df)
                
                with predc2:
                    with pred2c2:
                        if st.button("다중 CTQ 예측"):
                            result_df = pd.DataFrame(model.predict(ctp_df))
                            result_df.columns = Selected_y
                            st.session_state[pp3][1] = result_df
                            st.experimental_rerun()
                        
                        st.write("")
                        if st.session_state[pp3][1] is not None:
                            st.markdown("**▼ 다중 CTQ 예측 결과**")
                            st._legacy_dataframe(st.session_state[pp3][1])
                            st.write("")
                            st.write("")
                            st.markdown('**· 다중 CTQ 예측 결과 다운로드**')
                            total_df = pd.concat([st.session_state[pp3][0],st.session_state[pp3][1]], axis=1)
                            st_pandas_to_csv_download_link(total_df, file_name = "Predicted_CTQs_Results.csv", itx="Predicted_CTQs_Results.csv")

        
























