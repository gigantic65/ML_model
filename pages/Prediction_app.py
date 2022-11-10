
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import random

#st.set_page_config(page_title='Prediction_app')


def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataframe (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)
    
    
def app():
    
    st.markdown("<h5 style='text-align: right; color: black;'>KCC Ver. 1.0</h5>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: black;'>Machine Learning Prediction</h2>", unsafe_allow_html=True)

    st.write("")

    
    with st.expander("Predict New Conditions Guide"):
        st.write(
                "1. 저장된 Trained Data 불러오기.\n"
                "2. 저장된 Trained Model 불러오기.\n"
                "3. 1).Single case prediction    : 하나의 조건에 대해 직접 입력 후 예측.\n"
                "3. 2).Multiple case predicition : 여러 조건들을 CVS 파일로 저장 후 파일을 불러와서 예측.\n"
        )


    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload Train Data File'):
        uploaded_file = st.sidebar.file_uploader( 'Train Data File', type=["csv"])
  
    #---------------------------------#
    # Main panel

    # Displays the dataset

    # Displays the dataset
    st.subheader('**1. Uploaded Train data**')
    st.write('')

    if uploaded_file is not None:
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        df = load_csv()
    
        
        #x = list(df.columns[:-1])
        #y = list(df.columns[df.shape[1]-1:])
        
        x = list(df.columns)
        #y = list(df2.columns[df2.shape[1]-2:])

        Selected_X = st.sidebar.multiselect('X variables', x, x)
        
        y = [a for a in x if a not in Selected_X]
        
        Selected_y = st.sidebar.multiselect('Y variables', y, y)

        #Selected_X = st.sidebar.multiselect('X variables', x, x)
        #Selected_y = st.sidebar.multiselect('Y variables', y, y)
            
        Selected_X = np.array(Selected_X)
        Selected_y = np.array(Selected_y)
        
        st.write('**1.1 Number of X Variables:**',Selected_X.shape[0])
        st.info(list(Selected_X))
        st.write('')
    
        st.write('**1.2 Number of Y Variables:**',Selected_y.shape[0])
        st.info(list(Selected_y))
    
        df2 = pd.concat([df[Selected_X],df[Selected_y]], axis=1)
        #df2 = df[df.columns == Selected_X]
    
        #Selected_xy = np.array((Selected_X,Selected_y))

        

        st.write('')
        st.write('')
        st.write('')
    
        
        
    
        with st.sidebar.header('2. Upload Train Model File'):
            uploaded_file2 = st.sidebar.file_uploader("Trained model file", type=["pkl"])
            
        st.subheader('**2. Uploaded Machine Learning Model**')
        st.write('')

        if uploaded_file2 is not None:
            def load_model(model):
                loaded_model = pickle.load(model)
                return loaded_model
            

            model = load_model(uploaded_file2)
            
        
            
            st.write('**2.1 Trained Machine Learning Model :**')
            st.write(model)
            st.write('')
            
        
            st.write('**2.2 Trained Model Accuracy :**')
            X_train = df[Selected_X]
            y_train = df[Selected_y]
            
            scaler_y = StandardScaler().fit(y_train)
    
            #rescaled = scaler.transform(X_train)
            
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
            #    names.append(name)
            msg.append('%s' % model)
            mean.append('%f' %  (cv_results.mean()))
            std.append('%f' % (cv_results.std()))
                    
                    
            F_result3 = pd.DataFrame(np.transpose(msg))
            F_result3.columns = ['Machine_Learning_Model']
            F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
            F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
            
            #st.write(F_result3)    

            st.write('Final Model Accuracy ($R^2$):')
            
            R2_mean = list(F_result3['R2_Mean'].values)
            st.info( R2_mean[0] )
                
            st.write('Model Accuracy Deviation (Standard Deviation):')
            
            R2_std = list(F_result3['R2_Std'].values)
            st.info( R2_std[0])
                
            
            
            
                        
            
            df2 = df[Selected_X]
            columns = df2.columns
    
            test = []
            name = []
            #st.sidebar.write('3.1 Predict Single Condition')
            
    
    
            if Selected_y.shape[0] <= 1:
                
                st.sidebar.header('3. Model Prediction')
                st.sidebar.write('3.1 Predict Single Condition')
                
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')

                for column in columns:
                        value = st.sidebar.number_input(column, 0.00, 1000.00, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                
                        name.append(column)
                        test.append(value)
                
 

                
                st.write('')
                st.write('')
                st.write('')        
                              
                st.subheader('**3. Model Prediction**')
                st.write('**3.1 Single Condition Prediction :**')
                
                F_result = pd.DataFrame(name)
                F_result.columns = ['X Variables']
                F_result['Value'] = pd.DataFrame(test)

                
                #st.write(F_result)
     
            #para = st.sidebar.slider(para,mi,ma,cu,inter)
                st.write('')
            
            
        
                st.write(F_result)
                #scaler = StandardScaler().fit(X_train)
            
                if st.sidebar.button('Run Prediction'):
            
            #st.write(F_result)
                    F_result = F_result.set_index('X Variables')
                    F_result_T = F_result.T
    
                    #rescaled2 = scaler.transform(F_result_T)
        
                    predictions = model.predict(F_result_T)
        
        
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> * Single Condition Results </h6>", unsafe_allow_html=True)
    
                    st.write('**Predicted_', Selected_y[0],"  :**" , predictions[0])
                
 
                
                
                
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('3.2 Predict Optimizing Condition')
            
            #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                st.sidebar.write('**Target Value :**')
                N_sample = 0
                Target = st.sidebar.number_input(Selected_y[0], 0.00, 1000.00, 0.00,format="%.2f")
            
                st.sidebar.write('**Random Sample Generator :**')
                N_sample = st.sidebar.number_input('Sample Number',0,1000,50, format="%d")
                
                name2=[]
                test2=[]
                count = 0
            
                st.sidebar.write('Process Condition Ranges')
                for column in columns:
                
                    max1 = round(float(df[column].max()),3)
                    min1 = round(float(df[column].min()),3)
                
                    rag1 = round(min1+((max1-min1)*0.1),3)
                    rag2 = round(min1+((max1-min1)*0.9),3)
                
                    step = round((max1-min1)/20.0,3)
                
                    value = st.sidebar.slider(column, min1, max1, (rag1,rag2), step)
                         
                    name2.append(column)
                    test2.append(value)
                #param2.append(para_range)
                #st.write(min1,rag1,rag2,max1)
                #st.write(column)
                #st.write(test2)
            
                st.write('')
                st.write('**3.2 Optimizing Condition Prediction:**')
        
                if st.sidebar.button('Run Prediction',key = count): 
                
                    para = []
                    para2 = []
                    para4 = []
                
                    #st.write(test2)
                    import itertools
                
                    for para in test2:
                        if para[0] == para[1]:
                            para = itertools.repeat(para[0],100)
                        else:
                            para = np.arange(round(para[0],3), round(para[1]+((para[1]-para[0])/100.0),3), round((para[1]-para[0])/100.0,3))
                    #st.write(para)
                        para2.append(para)
                
               
                    Iter = N_sample
                
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    para2 = para2.dropna().reset_index()
                    para2.drop(['index'],axis=1,inplace=True)
                
                    Iter2 = para2.shape[1]
                
                

                #st.write(Iter,Iter2)
                    for i in range(Iter):
                        para3 = []
                        para5 = []
                        for j in range(Iter2):
                            #st.write(i,j,list(para2[j]))
                            para3.append(random.sample(list(para2[j]),1))
                            para5.append(para3[j][0])
                        
                    #para3 = pd.DataFrame(para3).values
                    #para4.append(para3)
                        para4.append(para5)
                    
                    
                #para4 = pd.DataFrame(para4)
                    para4 = pd.DataFrame(para4)
                
                
                    para4.columns=list(Selected_X)
                
                    st.write('**Selected Process Condtions:**')
                    st.write(para4)
                
                    datafile = para4
    
                    #rescaled = scaler.transform(datafile)
            
                    predictions2 = model.predict(datafile)
    
                    para4['predicted results'] = predictions2
                
                #st.write(para4)
                
                    para4.sort_values(by='predicted results', ascending=True, inplace =True)
                
                    para4 = para4.reset_index()
                    para4.drop(['index'],axis=1,inplace=True)
                
                #st.write(para4)
                
                    def find_nearest(array, value):
                        array = np.asarray(array)
                        idx = (np.abs(array - value)).argmin()
                        return array[idx]
                
                    opt_result = find_nearest(para4['predicted results'],Target)
                
                
                #st.write(opt_result)
    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)
                
                
                    df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                    df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                    df_opt = para4[para4['predicted results']==opt_result]
                    st.write('**Optimize Process Conditions:**')
                    st.write(df_opt)
                    st.write('**Maximize Process Conditions:**')
                    st.write(df_max)
                    st.write('**Minimize Process Conditions:**')
                    st.write(df_min)
                    #st.info(list(Selected_X2))
                
                    st.write('')
                    st.write('**Total results:**')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(para4)
                
         
                

                    rescaled = pd.DataFrame(datafile)
                
                #st_shap(shap.force_plot(explainer.expected_value, shap_values[:, :], rescaled.iloc[:, :]), height=300, width=700)
                
                
                
                    ax = sns.scatterplot(x=para4.index,y=para4.iloc[:,-1],s=30,color='blue')
                    sns.lineplot(x=para4.index,y=Target,ax=ax.axes, color='red')
                
                                
                
                    st.pyplot()
                
                
                    count +=1

    
                    st.markdown('**Download Predicted Results for Multi Conditions**')
        
                    st_pandas_to_csv_download_link(para4, file_name = "Predicted_Results.csv")
                    st.write('*Save directory setting : right mouse button -> save link as') 


            
        
        
                st.sidebar.write('')
                st.sidebar.write('')    
            
        
                st.sidebar.write('')
                st.sidebar.write('3.3 Predict multiple conditions data File')
                uploaded_file3 = st.sidebar.file_uploader("Upload file", type=["csv"])
                if uploaded_file3 is not None:
                    def load_csv():
                        csv = pd.read_csv(uploaded_file3)
                        return csv
                
    
    
#    uploaded_file = st.file_uploader("Choose a file")
#if uploaded_file is not None:
#    uploaded_file.seek(0)
#    data = pd.read_csv(uploaded_file, low_memory=False)
#    st.write(data.shape)
        
                count +=1
                st.write('**3.3 Multiple Conditions Prediction :**')
            
                if st.sidebar.button('Run Prediction',key = count):
                    df3 = load_csv()            
                    datafile = df3
    
                    #rescaled = scaler.transform(datafile)
                    
                    predictions2 = model.predict(datafile)
    
                    df3['predicted results'] = predictions2
    
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> * Multi Condition Results </h6>", unsafe_allow_html=True)
                
                    df_max = df3[df3['predicted results']==df3['predicted results'].max()]
                    df_min = df3[df3['predicted results']==df3['predicted results'].min()]
                    st.write('**Maximize Process Conditions:**')
                    st.write(df_max)
                    st.write('**Minimize Process Conditions:**')
                    st.write(df_min)
                    #st.info(list(Selected_X2))
                
                    st.write('')
                    st.write('**Total results:**')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(df3)
                    sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='red')
                    st.pyplot()
                    count +=1
        
    
    
                    st.markdown('**Download Predicted Results for Multi Conditions**')
        
                    st_pandas_to_csv_download_link(df3, file_name = "Predicted_Results.csv")
                    st.write('*Save directory setting : right mouse button -> save link as') 

            #st.write(mi)
        #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  
        
        
            else :
                
                st.sidebar.header('3. Model Prediction')
                st.sidebar.write('3.1 Predict Single Condition')
                
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')

                for column in columns:
                        value = st.sidebar.number_input(column, 0.00, 1000.00, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                
                        name.append(column)
                        test.append(value)
                
 

                
                st.write('')
                st.write('')
                st.write('')        
                              
                st.subheader('**3. Model Prediction**')
                st.write('**3.1 Single Condition Prediction :**')
                
                F_result = pd.DataFrame(name)
                F_result.columns = ['X Variables']
                F_result['Value'] = pd.DataFrame(test)

                
                #st.write(F_result)
     
            #para = st.sidebar.slider(para,mi,ma,cu,inter)
                st.write('')
            
            
        
                st.write(F_result)
                
            
                if st.sidebar.button('Run Prediction'):
            
            #st.write(F_result)
                    F_result = F_result.set_index('X Variables')
                    F_result_T = F_result.T
    
      
                    predictions = model.predict(F_result_T)
        

                    #predictions = pd.DataFrame(predictions[0],columns=['Value'])
                    #st.write(predictions)
                    
                    predictions2 = pd.DataFrame()
                    predictions2['Y Variable'] = df[Selected_y].columns
                    predictions2['Value'] = pd.DataFrame(predictions[0])
                    
                    
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> * Single Condition Results </h6>", unsafe_allow_html=True)
    
                    st.write('**Predicted_', Selected_y[0],"  :**" , predictions2)
                
 
                
                
                
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('3.2 Predict Optimizing Condition')
            
            #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                st.sidebar.write('**Target Value :**')
                
                sample_target = df[Selected_y].shape[1]
                
                #st.write(sample_target)
                
                Target = []
                
                
                for i in range(sample_target):
                    Target.append(st.sidebar.number_input(Selected_y[i], 0.00, 1000.00, 0.00,format="%.2f"))
            
            
            
                N_sample = 0
                st.sidebar.write('**Random Sample Generator :**')
                N_sample = st.sidebar.number_input('Sample Number',0,1000,50, format="%d")
                
                
                
                name2=[]
                test2=[]
                count = 0
            
                st.sidebar.write('Process Condition Ranges')
                for column in columns:
                
                    max1 = round(float(df[column].max()),3)
                    min1 = round(float(df[column].min()),3)
                
                    rag1 = round(min1+((max1-min1)*0.1),3)
                    rag2 = round(min1+((max1-min1)*0.9),3)
                
                    step = round((max1-min1)/20.0,3)
                
                    value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                         
                    name2.append(column)
                    test2.append(value)
                #param2.append(para_range)
                #st.write(min1,rag1,rag2,max1)
                #st.write(column)
                #st.write(test2)
            
                st.write('')
                st.write('**3.2 Optimizing Condition Prediction:**')
        
                if st.sidebar.button('Run Prediction',key = count): 
                
                    para = []
                    para2 = []
                    para4 = []
                
                    #st.write(test2)
                    import itertools
                
                    for para in test2:
                        if para[0] == para[1]:
                            para = itertools.repeat(para[0],100)
                        else:
                            para = np.arange(round(para[0],3), round(para[1]+((para[1]-para[0])/100.0),3), round((para[1]-para[0])/100.0,3))
                    #st.write(para)
                        para2.append(para)
                
               
                    Iter = N_sample
                
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    para2 = para2.dropna().reset_index()
                    para2.drop(['index'],axis=1,inplace=True)
                
                    Iter2 = para2.shape[1]
                
                

                #st.write(Iter,Iter2)
                    for i in range(Iter):
                        para3 = []
                        para5 = []
                        for j in range(Iter2):
                            #st.write(i,j,list(para2[j]))
                            para3.append(random.sample(list(para2[j]),1))
                            para5.append(para3[j][0])
                        
                    #para3 = pd.DataFrame(para3).values
                    #para4.append(para3)
                        para4.append(para5)
                    
                    
                #para4 = pd.DataFrame(para4)
                    para4 = pd.DataFrame(para4)
                
                
                    para4.columns=list(Selected_X)
                
                    st.write('**Selected Process Condtions:**')

                    st.write(para4)
                    
                    datafile = para4
              
                    predictions2 = model.predict(datafile)
                    
                    predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
                
                    
                    para4 = pd.concat([para4, predictions2],axis=1)
                        

                
                    #para4.sort_values(by='predicted results', ascending=True, inplace =True)
                
                    para4 = para4.reset_index()
                    para4.drop(['index'],axis=1,inplace=True)
                
                    
                    predictions3 = scaler_y.transform(predictions2)
                    
                    Target = pd.DataFrame(Target)
                    Target = Target.T
                    
                    Target2 = scaler_y.transform(Target)
                    
                    #iff3 = predictions3 - Target3
                    
                    #st.write(predictions2)
                    #st.write(Target)
                    #st.write(predictions3)
                    #st.write(Target2)
                    Target2 = pd.DataFrame(Target2)
                    
                    Target3 = []
                    
                    for i in range(Iter):
                        Target3.append(Target2.values)
                    
                    Target3 = np.reshape(Target3, (Iter, 3))
                    
                    Target3 = pd.DataFrame(Target3)
                    
                    #st.write(Target3)
                    
                    Diff3 = abs(predictions3 - Target3)
                    
                    #st.write(Diff3)

                    para4['sum'] = Diff3.sum(axis=1)
                    
                    para4.sort_values(by='sum', ascending=True, inplace =True)
                    
                    #st.write(para4)
                    
                    """
                    def find_nearest(array, value):
                        
                        array = np.asarray(array)
                        idx = (np.abs(array - value)).argmin()
                        return array[idx]
                
                    opt = predictions2.shape[1]
                    
                    opt_result = []
                    for i in range(opt):
                        
                        opt_result.append(find_nearest(predictions2.iloc[:,i],Target[i]))
  
    
                    st.write(opt_result)
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)
                
                
                    #df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                    #df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                    df_opt = pd.DataFrame()
                    
                    for i in range(opt):   
                        for column in df[Selected_y].columns:
                            df_opt = pd.concat([df_opt, para4[para4[column]==opt_result[i]]],axis=0)
                    """        
                    
                    st.write('**Optimize Process Conditions:**')
                    opt = para4.drop(['sum'],axis=1)
                    st.write(opt.head(5))

                
                    st.write('')
                    st.write('**Total results:**')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    
                    Target5 = []
                    for i in range(Iter):
                        Target5.append(Target.values)
                    Target5 = np.reshape(Target5, (Iter, 3))
                    Target5 = pd.DataFrame(Target5)
                    
                    plt.figure(figsize=(10,6))
                    fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                    fig.subplots_adjust(hspace=1)
                

                    for i in range(1,df[Selected_y].shape[1]+1):
                
                        plt.subplot(1,df[Selected_y].shape[1],i)
                        
                        sns.lineplot(x=para4.index, y=Target5[i-1], color='red')
                        
                        #plt.plot(df[Selected_y].iloc[:,i-1], df[Selected_y].iloc[:,i-1], color='blue', label = 'Actual data')
                        plt.scatter(para4.index, predictions2.iloc[:,i-1], color='blue', label = 'Prediction')
                        plt.title(df[Selected_y].columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    #ax.set_xlabel('Time', fontsize=16)
                    #plt.ylabel(Y.columns[i-1], fontsize=10)
                    
                    st.pyplot()
                    
                    st.write(opt)

                    
                
                    #ax = sns.scatterplot(x=para4.index,y=para4.iloc[:,-1],s=30,color='blue')
                    #sns.lineplot(x=para4.index,y=Target,ax=ax.axes, color='red')
                
                                
                
                    #st.pyplot()
                
                
                    count +=1

    
                    st.markdown('**Download Predicted Results for Multi Conditions**')
        
                    st_pandas_to_csv_download_link(para4, file_name = "Predicted_Results.csv")
                    st.write('*Save directory setting : right mouse button -> save link as') 


            
        
        
                st.sidebar.write('')
                st.sidebar.write('')    
            
        
                st.sidebar.write('')
                st.sidebar.write('3.3 Predict multiple conditions data File')
                uploaded_file3 = st.sidebar.file_uploader("Upload file", type=["csv"])
                if uploaded_file3 is not None:
                    def load_csv():
                        csv = pd.read_csv(uploaded_file3)
                        return csv
                
    
    
#    uploaded_file = st.file_uploader("Choose a file")
#if uploaded_file is not None:
#    uploaded_file.seek(0)
#    data = pd.read_csv(uploaded_file, low_memory=False)
#    st.write(data.shape)
        
                count +=1
                st.write('**3.3 Multiple Conditions Prediction :**')
            
                if st.sidebar.button('Run Prediction',key = count):
                    df3 = load_csv()            
                    datafile = df3
    
                    
                    predictions2 = model.predict(datafile)
    
                    predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
    
                    #df3['predicted results'] = predictions2
                    
                    predictions3 = pd.DataFrame()
                    predictions3 = pd.concat([df3, predictions2],axis=1)
                    
                    st.write(predictions3)
    
    
                    st.markdown("<h6 style='text-align: left; color: darkblue;'> * Multi Condition Results </h6>", unsafe_allow_html=True)
                
                    plt.figure(figsize=(10,6))
                    fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                    fig.subplots_adjust(hspace=1)
                
                    for i in range(1,df[Selected_y].shape[1]+1):
                

                    
                        plt.subplot(1,df[Selected_y].shape[1],i)
                        
                        #sns.lineplot(x=df3.index,y=Target[i-1], color='red')
                        
                        #plt.plot(df[Selected_y].iloc[:,i-1], df[Selected_y].iloc[:,i-1], color='blue', label = 'Actual data')
                        plt.scatter(df3.index, predictions2.iloc[:,i-1], color='red', label = 'Prediction')
                        plt.title(df[Selected_y].columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    #ax.set_xlabel('Time', fontsize=16)
                    #plt.ylabel(Y.columns[i-1], fontsize=10)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    
                
                    #st.write('')
                    #st.write('**Total results:**')
                    #st.set_option('deprecation.showPyplotGlobalUse', False)
                    #st.write(df3)
                    #sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='red')
                    #st.pyplot()
                    count +=1
        
    
    
                    st.markdown('**Download Predicted Results for Multi Conditions**')
        
                    st_pandas_to_csv_download_link(df3, file_name = "Predicted_Results.csv")
                    st.write('*Save directory setting : right mouse button -> save link as') 

            #st.write(mi)
        #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  
  