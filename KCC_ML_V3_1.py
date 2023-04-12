import streamlit as st
import pandas as pd
import time
import base64

from Stages import Stage1, Stage2, Stage3

st.set_page_config(page_title="KCC 머신러닝 프로그램",
                   page_icon="🧊",
                   layout="wide",
                   initial_sidebar_state="collapsed"
                   )

st.markdown("<h2 style='text-align: center; background-color:#0e4194; color: white;'>머신러닝 데이터분석 프로그램</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; background-color:#0e4194; color: white;'>[Machine Learning Application]</h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: right; color: black;'>KCC Ver. 3.1</h5>", unsafe_allow_html=True)

#=========================================================================================================================================================================

# 로그인
login = time.strftime("%Y%m%d%H%M%S")
if 'login' not in st.session_state: st.session_state['login'] = login

stage = 'stage_' + st.session_state['login']
if stage not in st.session_state: st.session_state[stage] = None

output_data = 'output' + st.session_state['login'] + '_1'
if output_data not in st.session_state: st.session_state[output_data] = pd.DataFrame()

output_data2 = 'output' + st.session_state['login'] + '_2'
if output_data2 not in st.session_state: st.session_state[output_data2] = [pd.DataFrame(),None,None]

#=========================================================================================================================================================================

# 프로그램  실행

stage_list = ["Stage1. 데이터파일 전처리하기",
              "Stage2. 머신러닝 모델 생성하기",
              "Stage3. 최적 조건 예측하기"]

if st.session_state[stage] is None:
    col1,col2 = st.columns(2)
    
    with col1:
        st.image('./pictures/smartfactory1.jpg',use_column_width='always')
    
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.subheader('▶ 시작 Stage 선택')
        
        st.write("")
        stage_selected = st.selectbox("※ 아래 목록에서 시작할 Stage를 선택하세요", stage_list)
        
        st.write("")
        if st.button("▶ Stage 시작"):
            st.session_state[stage] = stage_selected
            
            history = pd.read_csv("history.csv")
            history.loc[len(history),"login"] = login
            history.to_csv("history.csv", index=False)
            
            st.experimental_rerun()
        
        with open("프로그램 매뉴얼_V3_1.pptx", 'rb') as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            st.markdown(f'<a href="data:application/vnd.ms-powerpoint;base64,{b64}" download="프로그램 매뉴얼_V3_1.pptx">※매뉴얼 다운로드</a>', unsafe_allow_html=True)
        
    col21,col22,col23 = st.columns([2,1,3])
    
    with col22: st.caption("Photo by 작가 jcomp 출처 Freepik")

else:
    if len(st.session_state[output_data]) > 0:
        st.session_state[stage] = stage_list[1]
    
    if len(st.session_state[output_data2][0]) > 0:
        st.session_state[stage] = stage_list[2]
        
    if st.session_state[stage] == stage_list[0]:
        Stage1.app(st.session_state['login'])
        
    elif st.session_state[stage] == stage_list[1]:
        Stage2.app(st.session_state['login'])
        
    elif st.session_state[stage] == stage_list[2]:
        Stage3.app(st.session_state['login'])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
