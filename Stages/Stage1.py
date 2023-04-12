import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

import base64
import os.path

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
    with st.expander("Stage1. 데이터파일 전처리하기 - 가이드"):
        st.markdown("**1. 데이터 입력**")
        st.markdown("- 입력 방법 1: 엑셀 데이터를 복사 후 입력 영역에 붙여넣고 버튼을 Click 합니다.")
        st.markdown("- ※ 입력 양식 확인은 우측 양식 다운로드하여 확인할 수 있습니다.")
        st.markdown("- 입력 방법 2: 우측의 예제 데이터를 선택하여 진행합니다.")
        st.write("")
        st.markdown("**1.1. 입력 데이터 세트**")
        st.markdown("- 입력된 데이터 및 현황을 확인하고, 고유값이 1개인 열은 제거합니다.")
        st.write("")
        st.markdown("**2. 데이터 가공**")
        st.markdown("- 결측데이터, 반복데이터, 범주데이터, 이상데이터 발생 시 순서대로 데이터를 수정합니다.")
        st.markdown("- 데이터 가공 완료 후 Stage1. 데이터파일 전처리 완료 버튼을 Click 합니다.")
        st.markdown("- 가공된 데이터는 자동 저장되며, 다운로드 링크를 활용하여 별도 저장할 수 있습니다.")
        st.markdown("- Go to Stage2 버튼을 Click하여 다음 Stage로 이동합니다.")
    
    #사용자가 접속한 시간'session_in'을 session_state로 저장한 후 본 앱에 사용할 변수 선언
    d10 = 'df' + session_in + '_0'
    if d10 not in st.session_state: st.session_state[d10] = None
    
    d11 = 'df' + session_in + '_1'
    if d11 not in st.session_state: st.session_state[d11] = None
    
    ms_in = 'missing' + session_in + '_0'
    if ms_in not in st.session_state: st.session_state[ms_in] = False
    
    ms_chk = 'missing' + session_in + '_1'
    if ms_chk not in st.session_state: st.session_state[ms_chk] = False
    
    ol_in = 'outlier' + session_in + '_0'
    if ol_in not in st.session_state: st.session_state[ol_in] = False
    
    ol_chk = 'outlier' + session_in + '_1'
    if ol_chk not in st.session_state: st.session_state[ol_chk] = False
    
    output_data = 'output' + session_in + '_1'
    if output_data not in st.session_state: st.session_state[output_data] = pd.DataFrame()
    
#=========================================================================================================================================================================
    
    st.write("")
    #st.markdown("<h3 style='text-align: left; color: black;'>Stage1. 데이터파일 전처리하기</h3>", unsafe_allow_html=True)
    tic1,tic2,tic3 = st.columns(3)
    with tic1: st.image('./pictures/Stage1_c.png',use_column_width='always')
    with tic2: st.image('./pictures/Stage2_m.png',use_column_width='always')
    with tic3: st.image('./pictures/Stage3_m.png',use_column_width='always')
    st.write("")
    st.write("")
    
    st.subheader('1. 데이터 입력')
    
    lv1c1,lv1c2,lv1c3 = st.columns([4,1,1])
    
    with lv1c1:
        txt = st.text_area("엑셀 데이터영역 붙여넣기")
        if st.button("엑셀 데이터영역 붙여넣은 후 Click"):
            paste_res = paste_seq(txt)
            if paste_res is not None:
                st.session_state[d10] = paste_res
                st.session_state[d11] = paste_res
                st.session_state[ms_in] = False
                st.session_state[ms_chk] = False
                st.session_state[ol_in] = False
                st.session_state[ol_chk] = False
            st.experimental_rerun()
    
    with lv1c2:
        example = st.selectbox("예제 데이터",('Select ▼','예제1', '예제2'))
        if example == '예제1': st.session_state[d10] = pd.read_csv('./examples/example1.csv')
        elif example == '예제2': st.session_state[d10] = pd.read_csv('./examples/example2.csv')
    
    with lv1c3:
        st.markdown("<p style='font-size:14px'>양식 다운로드</p>", unsafe_allow_html=True)
        st_pandas_to_csv_download_link(pd.read_csv('./examples/example1.csv'), file_name = "example1.csv", itx="example1.csv")
    #버튼 데이터와 예제 데이터 변경 시 이상 > selectbox 초기화 방법 있을까?
    
    st.write("")
    st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)
    st.write("")
    
#=========================================================================================================================================================================

###### Main panel ##########

# Displays the dataset
    if st.session_state[d10] is not None:
        df = st.session_state[d10]
        
        lv2c1,lv2c2 = st.columns(2, gap="large")
        
        with lv2c1:
            st.markdown('**1.1 입력 데이터 세트(Data Set)**')
            st.caption("**행 : 케이스(Sample)  /  열 : 특성(Feature)**")
            st._legacy_dataframe(df)
        
        with lv2c2:
            st.markdown('**1.2 입력 데이터 현황(Statistics)**')
            st.caption("**공정 인자 별 특성 확인**")
            
            st._legacy_dataframe(df.describe()) #statistics확인
        
        st.write("")
        st.markdown("""<hr style="height:2px;border:none;color:rgb(60,90,180); background-color:rgb(60,90,180);" /> """, unsafe_allow_html=True)
        st.write("")
        
        st.subheader('2. 데이터 가공')
        st.markdown('**- 결측데이터, 반복데이터, 범주데이터, 이상데이터 처리**')
        
        if st.session_state[d11] is None: st.session_state[d11] = df.copy()
        df11 = st.session_state[d11]
        
        errcl1, errcl2, errcl3, errcl4, errcl_gap, errcl5 = st.columns([4,4,4,4,1,16])
        all_chk = [False,False,False,False]
        
        with errcl1:
            #st.write('**2.3 결측 데이터(Missing Data) 확인**')
            miss = df11.isnull().sum().sum()
            if miss == 0:
                st.success("결측데이터 없음")
                all_chk[0] = True
            elif miss != 0:
                if st.session_state[ms_chk] == False:
                    if st.button('결측데이터 존재 ▶Click!!!!!◀'):
                        st.session_state[ms_in] = True
                        st.session_state[ms_chk] = True
                        st.experimental_rerun()
                else: st.warning("**▼결측데이터 처리중**")
        
        with errcl2:
            #st.write('**2.2 반복 데이터(Duplicated Data) 확인**')
            dupli = df11.duplicated().sum()
            if dupli == 0:
                st.success("반복데이터 없음")
                all_chk[1] = True
            else:
                if False not in all_chk[:1]:
                    if st.button('반복데이터 존재 ▶Click!!!!!◀'):
                        df11 = df11.drop_duplicates()
                        st.session_state[d11] = df11
                        st.experimental_rerun()
                else: st.error("반복데이터 존재")
        
        with errcl3:
            #st.markdown('**2.1 범주 데이터(Categorical Data) 확인**')
            cla = 0
            for column in df11.columns:
                if df11[column].dtypes == 'O':
                    cla = 1
                    break
                        
            if cla == 0:
                st.success("범주데이터 없음")
                all_chk[2] = True
            else:
                if False not in all_chk[:2]:
                    if st.button('범주데이터 존재 ▶Click!!!!!◀'): 
                        le = preprocessing.LabelEncoder()
                        for column in df11.columns:
                            if df11[column].dtypes == 'O':
                                df11[column] = le.fit_transform(df11[column])
                        st.session_state[d11] = df11
                        st.experimental_rerun()
                else: st.error("범주데이터 존재")
        
        with errcl4:
            if st.session_state[ol_chk] == True:
                st.success("이상데이터 없음")
                all_chk[3] = True
            elif st.session_state[ol_in] == False:
                if False not in all_chk[:3]:
                    if st.button('이상데이터 확인 ▶Click!!!!!◀'):
                        st.session_state[ol_in] = True
                        st.experimental_rerun()
                else: st.error("이상데이터 확인")
            else: st.warning("**▼이상데이터 처리중**")
        
        with errcl5:
            st.write("")
            st.markdown('**▼ 데이터 가공 결과**')
        
        lv3c1,lv3c2 = st.columns(2, gap="large")
        
        with lv3c1:
            if st.session_state[ms_in] == True:
                df11 = st.session_state[d11].copy()
                
                st.write("")
                st.write("")
                option = st.selectbox('결측값 처리 방법 선택',('select ▼','행 전체 제거하기','0(Zero)으로 채우기','앞의 값으로 채우기', '뒤의 값으로 채우기','보간 방법(Interpolation)으로 채우기','평균 값으로 채우기'))
                if option == '행 전체 제거하기': df11 = df11.dropna().reset_index(drop=True)
                elif option == '0(Zero)으로 채우기': df11 = df11.fillna(0)
                elif option == '앞의 값으로 채우기': df11 = df11.fillna(method='ffill')
                elif option == '뒤의 값으로 채우기': df11 = df11.fillna(method='bfill')
                elif option == '보간 방법(Interpolation)으로 채우기': df11 = df11.fillna(df11.interpolate())
                elif option == '평균 값으로 채우기': df11 = df11.fillna(df11.mean())
                
                if option != 'select ▼':
                    if st.button('결측값 처리'):
                        df11 = df11.apply(pd.to_numeric, errors='ignore')
                        st.session_state[d11] = df11
                        st.session_state[ms_in] = False
                        st.experimental_rerun()
            
            if st.session_state[ol_in] == True:
                #Feature(X인자)의 리스트 발췌 -> 버튼 만들 때 리스트로 사용
                df11 = st.session_state[d11].copy()
                
                st.write("")
                st.write("")
                st.markdown("**▼ 고유값 이상 데이터 확인**")
                
                unique_col = []
                for column in df11.columns:
                    if len(df11[column].unique()) == 1: unique_col.append(column)
                
                if len(unique_col) >= 1:
                    st.error("하나의 고유값을 가진 공정인자 %s 삭제 처리됨" %unique_col)
                    df11 = df11.drop(unique_col, axis=1)
                else: st.success("고유값 이상 데이터 없음")
                
                st.write("")
                st.write("")
                st.markdown("**▼ 경계(Nσ) 기준에 따른 이상 데이터 확인**")
                
                non_obj_list = []
                for column in df11.columns:
                    if df11[column].dtypes != 'O': non_obj_list.append(column)
                
                out2 = st.number_input('이상데이터 경계 선택하기(Number of Sigma(σ)) : ',0,10,3, format="%d")
                
                #np.nan 이 있지만, 실제 지우지는 않고 outlier개수 확인 용 코드
                df11_ = df11[non_obj_list].copy()
                for i in range(len(df11_.columns)):
                    df_std = np.std(df11_.iloc[:,i])
                    df_mean = np.mean(df11_.iloc[:,i])
                    cut_off = df_std * out2
                    upper_limit = df_mean + cut_off
                    lower_limit = df_mean - cut_off
                
                    for j in range(df11_.shape[0]):
                        if df11_.iloc[j,i] > upper_limit or df11_.iloc[j,i] <lower_limit: df11_.iloc[j,i] = np.nan
                            
                out1 = df11_.isna().sum().sum() #out2기준을 넘는 outlier 개수
                
                #Outlier Cut
                if out1 == 0:
                    st.success("경계(Nσ) 기준을 초과한 이상 데이터 없음")
                    if st.button('이상 데이터 확인 완료'):
                        st.session_state[ol_in] = False
                        st.session_state[ol_chk] = True
                        st.experimental_rerun()
                        
                else:
                    st.warning("경계(Nσ) 기준을 초과한 이상 데이터 발생")
                    dtemp = pd.DataFrame(df11_[df11_.isna().any()[df11_.isna().any() == True].index.tolist()].isnull().sum()).transpose()
                    dtemp.index = ["이상데이터_개수"]
                    st._legacy_dataframe(dtemp)
                    
                    option3 = st.selectbox('※이상데이터 처리 방법 선택',('select ▼','행 전체 제거하기','0(Zero)으로 채우기','앞의 값으로 채우기','뒤의 값으로 채우기','보간 방법(Interpolation)으로 채우기','평균 값으로 채우기','처리하지 않기'))
                    
                    if option3 == '행 전체 제거하기': df11 = df11_.dropna().reset_index(drop=True)
                    elif option3 == '0(Zero)으로 채우기': df11 = df11_.fillna(0)
                    elif option3 == '앞의 값으로 채우기': df11 = df11_.fillna(method='ffill')
                    elif option3 == '뒤의 값으로 채우기': df11 = df11_.fillna(method='bfill')
                    elif option3 == '보간 방법(Interpolation)으로 채우기': df11 = df11_.fillna(df11.interpolate())
                    elif option3 == '평균 값으로 채우기': df11 = df11_.fillna(df11.mean())
                    elif option3 == '처리하지 않기': pass
                    
                    if option3 != 'select ▼':
                        if st.button('이상 데이터 처리'):
                            st.session_state[d11] = df11
                            st.session_state[ol_in] = False
                            st.session_state[ol_chk] = True
                            st.experimental_rerun()
                
                newlist =  ['select ▼'] + non_obj_list 
                
                st.write("")
                st.write("")
                st.markdown("**▼ 데이터 특성 그래프로 확인하기**")
                out3 = st.selectbox("데이터 선택", newlist)
                
                #Outlier Plot
                if out3 in df11.columns:
                    OUT = st.session_state[d11][[out3]]
                    x = range(OUT.shape[0])
                    
                    OUT['max'] = np.mean(OUT)[0] + out2 * np.std(OUT)[0]  #Out = Outlier Criteria (σ)
                    OUT['min'] = np.mean(OUT)[0] - out2 * np.std(OUT)[0]
                    OUT['mean'] = np.mean(OUT)[0]
                    
                    sns.set(rc={'figure.figsize':(30,7)})
                   
                    plt.plot(x, OUT[out3], color = "black")
                    plt.plot(x, OUT['max'], color = "red", label='Outlier limit')
                    plt.plot(x, OUT['min'], color = "red")
                    plt.plot(x, OUT['mean'], color = "#0e4194", label='Mean Value')
                    plt.legend(loc='upper left', fontsize=15)
                    
                    st.pyplot(plt)
                
        with lv3c2:
            st._legacy_dataframe(st.session_state[d11])
            """
            # 데이터 타입 확인용 출력 : 열 중간에 텍스트가 존재하여 범주형으로 잡히는 경우 등
            st.write("")
            st.markdown("**▼ 데이터 타입**")
            st.write("")
            st._legacy_dataframe(pd.DataFrame(st.session_state[d11].dtypes).T)
            """
        with lv3c1:
            if False not in all_chk:
                if st.button('Stage1. 데이터파일 전처리 완료'):
                    st.session_state[output_data] = df11
                    
                    st.write("")
                    st.markdown('**_· Stage1 데이터 자동 저장 완료_**')
                    st.write("")
                    
                    if st.button("▶Go to Stage2"):
                        st.experimental_rerun()
                    
                    st.write("")
                    st.markdown('**_· 전처리 완료 데이터(.csv) 다운로드_**')
                    st_pandas_to_csv_download_link(df11, file_name = "Final_cleaned_data.csv")
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                    
