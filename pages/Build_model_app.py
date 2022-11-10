import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.metrics as sm
from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

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
import base64
import plotly.graph_objects as go
import pickle
import os.path






#---------------------------------#
# Model building




def feature_s(df3):
    
    x = df3.iloc[:,:-1] # Using all column except for the last column as X
    y = df3.iloc[:,-1] 
        

    r2_train = []
    r2_test = []
    
    for n_comp in range(x.shape[1]):
        
        n_comp += 1
        # define the method
        rfe = RFE(RandomForestRegressor(), n_features_to_select=n_comp)
        
        # fit the model
        rfe.fit(x, y)
        
        # transform the data
        X3 = rfe.transform(x)
        
        #   Column 값 넣기
        X_rfe = pd.DataFrame(X3)
        
        #st.write(X_rfe)
        
        X_rfe_columns = []
        
        for i in range(x.shape[1]):
            if rfe.ranking_[i] == 1:
                X_rfe_columns.append(x.columns[i])
        
        #for i in range(col):
        #    X_rfe_columns.append(x.columns[i])
            
            #    X_rfe.columns = X2_columns
        #st.write(X_rfe_columns)    
        X_train, X_test, y_train, y_test = train_test_split(X_rfe,
                                                            y, test_size=0.2, random_state=7)
            
        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        rescaledtestX = scaler.transform(X_test)

        model = RandomForestRegressor()
        #    MultiOutputRegressor(LinearRegression()).fit(rescaledX, y_train)
        model.fit(rescaledX, y_train)
        
        
        predictions = model.predict(rescaledX)
        predictions2 = model.predict(rescaledtestX)
             

        r_squared = sm.r2_score(y_train,predictions)
        r_squared2 = sm.r2_score(y_test,predictions2)
        
        #st.write(r_squared)
        
        
        r2_train.append(r_squared)
        r2_test.append(r_squared2)
        #st.write(X_rfe_columns)
    
    sns.set(rc={'figure.figsize':(10,5)})
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("** The importance of X variables**")
    feat_importances = pd.Series(model.feature_importances_, index=X_rfe_columns)
    feat_importances.plot(kind='barh')
    st.pyplot()
          
      
    NumX = r2_train.index(np.max(r2_train))+1
    NumX2 = r2_test.index(np.max(r2_test))+1
    st.write("** The model accuracy over the number of X-variables;   Optimum number: **", max(NumX,NumX2))
    #st.write("* Optimum number of the X variables = ", max(NumX,NumX2))
    
        
    sns.set(rc={'figure.figsize':(8,5)})
    plt.plot(range(1,x.shape[1]+1),r2_train, color="blue", marker= "o",label='Train')
    plt.plot(range(1,x.shape[1]+1),r2_test, color="red", marker= "o",label='Test')
    plt.xlabel('Number of X-variables')
    plt.ylabel('Model Accuracy (R2)')
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def F_feature(df3,hobby):
    
    x = df3.iloc[:,:-1] # Using all column except for the last column as X
    y = df3.iloc[:,-1]
    

        
    X_rfe_columns = []
    
        
    rfe = RFE(RandomForestRegressor(), n_features_to_select=hobby)
        
        # fit the model
    rfe.fit(x, y)
        
    for i in range(x.shape[1]):
        if rfe.ranking_[i] == 1:
            X_rfe_columns.append(x.columns[i])

    return X_rfe_columns



def feature_m(df3,Selected_X,Selected_y):
    
    x = df3[Selected_X] # Using all column except for the last column as X
    y = df3[Selected_y] 
        

    X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y, test_size=0.2, random_state=7)
        
        
    rfe = MultiOutputRegressor(RandomForestRegressor())
    
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

      
    # fit the model
    rfe.fit(rescaledX, y_train)
        
    f_importance = pd.DataFrame(rfe.estimators_[0].feature_importances_,columns=['importance'],index=X_train.columns)
    
    f_importance = f_importance.sort_values(by='importance', ascending = True)
    
    sns.set(rc={'figure.figsize':(10,5)})
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("** The importance of X variables**")
    
    f_importance.plot(kind='barh')
    
    st.pyplot()
    
    return f_importance

        
def F_feature_m(df3,hobby,Selected_X,Selected_y):
    
    x = df3[Selected_X] # Using all column except for the last column as X
    y = df3[Selected_y] 


    X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y, test_size=0.2, random_state=7)
        
        
    rfe = MultiOutputRegressor(RandomForestRegressor())
    
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)

      
    # fit the model
    rfe.fit(rescaledX, y_train)
        
    f_importance = pd.DataFrame(rfe.estimators_[0].feature_importances_,columns=['importance'],index=X_train.columns)
    
    f_importance = f_importance.sort_values(by='importance', ascending = False)
    

        
    X_rfe_columns = []
    
        
    for i in range(hobby):
        X_rfe_columns.append(f_importance.index[i])

    #st.write(X_rfe_columns)
    return X_rfe_columns
        

    

def build_model(df3,Selected_ml):
    
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y
    

        #test = st.multiselect('X variables',df.columns,df.colums)
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    st.markdown('**4.1. Model Validation Method:** K-Fold Cross Validation')
    st.markdown('**4.2. Machine Learning Model Comparison Results**')
    ml = pd.DataFrame(Selected_ml)
    #st.write(ml.values)
    
    i=0
    models = []
 
    
    #scaler = StandardScaler().fit(X)
    #rescaled = scaler.transform(X)

    
   
    for i in range(ml.shape[0]):
        if ml.iloc[i].values == 'Linear Regression':
            models.append(('Linear Regression', Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])))
                                                         
        if ml.iloc[i].values == 'Lasso':
            models.append(('LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
        if ml.iloc[i].values == 'KNN':
            models.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
            
        if ml.iloc[i].values == 'Decision_Tree':
            models.append(('Decision_Tree', Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())])))
            
        if ml.iloc[i].values == 'GBM':
            models.append(('GBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor(n_estimators=75))])))
        if ml.iloc[i].values == 'XGBOOST':
            models.append(('XGBOOST', Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(booster='gbtree',n_estimators= 100))])))
            
        if ml.iloc[i].values == 'AB':
            models.append(('AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
            
        if ml.iloc[i].values == 'Extra Trees':
            models.append(('Extra Trees', Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor())])))
        if ml.iloc[i].values == 'RandomForest':
            models.append(('RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor())])))

            
    results = []
    names = []

    msg = []
    mean = []
    max1 = []
    min1 = []
    std = []        
    for name, model in models:
        #    print(X_train, y_train)
        model = model
        
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
        
        for i, element in enumerate(cv_results):
            if element <= 0.0:
                cv_results[i] = 0.0
        

        results.append(abs(cv_results))
        
        names.append(name)
        #    names.append(name)
        msg.append('%s' % (name))
        mean.append('%f' %  (cv_results.mean()))
        min1.append('%f' %  (cv_results.min()))
        max1.append('%f' %  (cv_results.max()))
        std.append('%f' % (cv_results.std()))
        
        
    F_result = pd.DataFrame(np.transpose(msg))
    F_result.columns = ['Machine_Learning_Model']
    F_result['R2_Mean'] = pd.DataFrame(np.transpose(mean))
    F_result['R2_Min'] = pd.DataFrame(np.transpose(min1))
    F_result['R2_Max'] = pd.DataFrame(np.transpose(max1))
    F_result['R2_Std'] = pd.DataFrame(np.transpose(std))
  

    F2_result = F_result.sort_values(by='R2_Mean', ascending=False, inplace =False)

    st.markdown('*Number of Split Sample : 5, Scoring: ($R^2$)*')
    st.write(F_result)

    
    F_result['R2_Mean'] = F_result['R2_Mean'].astype('float')
    F_result['R2_Min'] = F_result['R2_Min'].astype('float')
    F_result['R2_Max'] = F_result['R2_Max'].astype('float')
    F_result['R2_Std'] = F_result['R2_Std'].astype('float')

    
    fig, axs = plt.subplots(ncols=2)
    g = sns.barplot(x="Machine_Learning_Model", y="R2_Mean", data=F_result, ax=axs[0])
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set(ylim=(0,1))
    
   
    g2 = sns.barplot(x="Machine_Learning_Model", y="R2_Std", data=F_result, ax=axs[1])
    g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
    g2.set(ylim=(0,1))
    
   

    st.pyplot(plt)
    
    return F2_result


def build_model_m(df3,Selected_ml,Selected_X,Selected_y):
    
    x = df3[Selected_X] # Using all column except for the last column as X
    y = df3[Selected_y] 
    
        #test = st.multiselect('X variables',df.columns,df.colums)
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    st.markdown('**4.1. Model Validation Method:** K-Fold Cross Validation')
    st.markdown('**4.2. Machine Learning Model Comparison Results**')
    ml = pd.DataFrame(Selected_ml)
    #st.write(ml.values)
    
    i=0
    models = []
 
    
    #scaler = StandardScaler().fit(X)
    #rescaled = scaler.transform(X)

    
   
    for i in range(ml.shape[0]):
        if ml.iloc[i].values == 'Linear Regression':
            models.append(('Linear Regression', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())]))))
                                                         
        if ml.iloc[i].values == 'Lasso':
            models.append(('LASSO', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())]))))
            
        if ml.iloc[i].values == 'KNN':
            models.append(('KNN', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())]))))
            
        if ml.iloc[i].values == 'Decision_Tree':
            models.append(('Decision_Tree', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())]))))
            
        if ml.iloc[i].values == 'GBM':
            models.append(('GBM', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('GBM',GradientBoostingRegressor(n_estimators=75))]))))
            
            
        if ml.iloc[i].values == 'XGBOOST':
            models.append(('XGBOOST', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(n_estimators= 100))]))))
            
        if ml.iloc[i].values == 'AB':
            models.append(('AB', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())]))))
            
        if ml.iloc[i].values == 'Extra Trees':
            models.append(('Extra Trees', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor())]))))
        if ml.iloc[i].values == 'RandomForest':
            models.append(('RandomForest', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor())])))) 

            
    results = []
    names = []

    msg = []
    mean = []
    max1 = []
    min1 = []
    std = []        
    for name, model in models:
        #    print(X_train, y_train)
        model = model
        
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring='r2')
        
        for i, element in enumerate(cv_results):
            if element <= 0.0:
                cv_results[i] = 0.0
        

        results.append(abs(cv_results))
        
        names.append(name)
        #    names.append(name)
        msg.append('%s' % (name))
        mean.append('%f' %  (cv_results.mean()))
        min1.append('%f' %  (cv_results.min()))
        max1.append('%f' %  (cv_results.max()))
        std.append('%f' % (cv_results.std()))
        
        
    F_result = pd.DataFrame(np.transpose(msg))
    F_result.columns = ['Machine_Learning_Model']
    F_result['R2_Mean'] = pd.DataFrame(np.transpose(mean))
    F_result['R2_Min'] = pd.DataFrame(np.transpose(min1))
    F_result['R2_Max'] = pd.DataFrame(np.transpose(max1))
    F_result['R2_Std'] = pd.DataFrame(np.transpose(std))
  

    F2_result = F_result.sort_values(by='R2_Mean', ascending=False, inplace =False)

    st.markdown('*Number of Split Sample : 5, Scoring: ($R^2$)*')
    st.write(F_result)

    
    F_result['R2_Mean'] = F_result['R2_Mean'].astype('float')
    F_result['R2_Min'] = F_result['R2_Min'].astype('float')
    F_result['R2_Max'] = F_result['R2_Max'].astype('float')
    F_result['R2_Std'] = F_result['R2_Std'].astype('float')

    
    fig, axs = plt.subplots(ncols=2)
    g = sns.barplot(x="Machine_Learning_Model", y="R2_Mean", data=F_result, ax=axs[0])
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set(ylim=(0,1))
    
   
    g2 = sns.barplot(x="Machine_Learning_Model", y="R2_Std", data=F_result, ax=axs[1])
    g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
    g2.set(ylim=(0,1))
    
   

    st.pyplot(plt)
    
    return F2_result
    
    
    
def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataframe (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)

def st_pandas_to_csv_download_link2(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataframe (CSV) </a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)


def Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid):
    
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y
    
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    
    
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)


    if Model == 'GBM':
        model = GradientBoostingRegressor(n_estimators=parameter_n_estimators, max_features=parameter_max_features)
    elif Model == 'Extra Trees':
        model = ExtraTreesRegressor(n_estimators=parameter_n_estimators, max_features=parameter_max_features)
    elif Model == 'RandomForest':
        model = RandomForestRegressor(n_estimators=parameter_n_estimators, max_features=parameter_max_features)
        

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
               

        
    grid.fit(rescaled, Y)
                

        
    st.write("**The best parameters are %s .**" % (grid.best_params_))
            

            #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    
    #-----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                        text='n_estimators')
                ),
        yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                        text='max_features')
                ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='Hyperparameter tuning results',
        scene = dict(
            xaxis_title='n_estimators',
            yaxis_title='max_features',
            zaxis_title='R2'),
            autosize=False,
            width=800, height=800,
            margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

        
    #st.write("The best parameters are %s ." % (grid.best_params_))
        
    return grid.best_params_['n_estimators'],grid.best_params_['max_features']



def Opti_model_m(Model,df3,param_grid,Selected_X2,Selected_y):
    
    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y
    
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)


    if Model == 'GBM':
        model = GradientBoostingRegressor()
    elif Model == 'Extra Trees':
        model = ExtraTreesRegressor()
    elif Model == 'RandomForest':
        model = RandomForestRegressor()
        

    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
               
        
    grid.fit(rescaled, Y)
                
    
    st.write("**The best parameters are %s .**" % (grid.best_params_))
    

    #st.write("The best parameters are %s ." % (grid.best_params_))
        
    return grid.best_params_['estimator__n_estimators'],grid.best_params_['estimator__max_features']





def Opti_model2(Model,df3,parameter_n_estimators,parameter_max_depth,param_grid):
    
    #X = df3.iloc[:,:-1] # Using all column except for the last column as X
    #Y = df3.iloc[:,-1] # Selecting the last column as Y
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y
    
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    
    
    scaler = StandardScaler().fit(X)

    

    model = xgboost.XGBRegressor(booster='gbtree',n_estimators=parameter_n_estimators, max_depth=parameter_max_depth)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)
                
    #st.subheader('Model Performance')
        
    y_pred_test = grid.predict(rescaled)
    #st.write('Coefficient of determination ($R^2$):')
    #st.info( r2_score(Y, y_pred_test) )
        
    #st.write('Error (MSE or MAE):')
    #st.info( mean_squared_error(Y, y_pred_test))
        
    st.write("**The best parameters are %s .**" % (grid.best_params_))
            

            #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['n_estimators','max_depth']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['n_estimators', 'max_depth', 'R2']
    grid_pivot = grid_reset.pivot('n_estimators', 'max_depth')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    
    #-----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                        text='max_depth')
                ),
        yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                        text='n_estimators')
                ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='Hyperparameter tuning',
        scene = dict(
            xaxis_title='max_depth',
            yaxis_title='n_estimators',
            zaxis_title='R2'),
            autosize=False,
            width=800, height=800,
            margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)
    
    
    return grid.best_params_['n_estimators'],grid.best_params_['max_depth']



def Opti_model2_m(Model,df3,param_grid,Selected_X2,Selected_y):
    
    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    
    
    scaler = StandardScaler().fit(X)

    

    model = xgboost.XGBRegressor()

    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)
                
    #st.subheader('Model Performance')
        

        
    st.write("**The best parameters are %s .**" % (grid.best_params_))
            

    
    return grid.best_params_['estimator__n_estimators'],grid.best_params_['estimator__max_depth']




def Opti_model3(Model,df3,parameter_n_estimators,parameter_learning_rate,param_grid):
    
    #X = df3.iloc[:,:-1] # Using all column except for the last column as X
    #Y = df3.iloc[:,-1] # Selecting the last column as Y
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y
    
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    
    
    scaler = StandardScaler().fit(X)

    

    model = AdaBoostRegressor(n_estimators=parameter_n_estimators, learning_rate=parameter_learning_rate)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)
                
    #st.subheader('Model Performance')
        
    y_pred_test = grid.predict(rescaled)
    #st.write('Coefficient of determination ($R^2$):')
    #st.info( r2_score(Y, y_pred_test) )
        
    #st.write('Error (MSE or MAE):')
    #st.info( mean_squared_error(Y, y_pred_test))
        
    st.write("**The best parameters are %s .**" % (grid.best_params_))
            

            #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['n_estimators','learning_rate']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['n_estimators', 'learning_rate', 'R2']
    grid_pivot = grid_reset.pivot('n_estimators', 'learning_rate')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    
    #-----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                        text='learning_rate')
                ),
        yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                        text='n_estimators')
                ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='Hyperparameter tuning',
        scene = dict(
            xaxis_title='learning_rate',
            yaxis_title='n_estimators',
            zaxis_title='R2'),
            autosize=False,
            width=800, height=800,
            margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)
    
    
    return grid.best_params_['n_estimators'],grid.best_params_['learning_rate']


def Opti_model3_m(Model,df3,param_grid,Selected_X2,Selected_y):
    
    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y
    
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    
    
    scaler = StandardScaler().fit(X)

    

    model = AdaBoostRegressor()

    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)
                

    st.write("**The best parameters are %s .**" % (grid.best_params_))
               
    
    return grid.best_params_['estimator__n_estimators'],grid.best_params_['estimator__learning_rate']


    


def Opti_KNN_model(df3, parameter_n_neighbors_knn,parameter_n_neighbors_step_knn,param_grid_knn):

    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y
    
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    
    
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)
        

    model = KNeighborsRegressor(n_neighbors=parameter_n_neighbors_knn)
        
    grid = GridSearchCV(estimator=model, param_grid=param_grid_knn, cv=5)
            
    grid.fit(rescaled, Y)
    
    st.write("**The best parameters is %s .**" % (grid.best_params_))
                

          
    
    return grid.best_params_['n_neighbors']


def Opti_KNN_model_m(df3, param_grid,Selected_X2,Selected_y):

    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y
    
    #st.write(X)
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    
    
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)
        

    model = KNeighborsRegressor()
        
    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
            
    grid.fit(rescaled, Y)
    
    st.write("**The best parameters is %s .**" % (grid.best_params_))
                

    return grid.best_params_['estimator__n_neighbors']






def download_model(k, model):
    
    if k==0:
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
        st.markdown(href, unsafe_allow_html=True)
    elif k==1:
        model.save('test.h5')
        st.write('Currently, Neural network model save is underway.')
        st.write('If you want to make Neural network model, please contact Simulation team.')



def app():
    st.markdown("<h5 style='text-align: right; color: black;'>KCC Ver. 1.0</h5>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: left; color: black;'>Build Machine Learning Model</h2>", unsafe_allow_html=True)
    

    st.write("")

 
            
    with st.expander("Machine Learning Application Guide"):
        st.write(
                "0. 공정변수를 X인자로 품질결과를 Y인자로 순서대로 나열해서 CSV 파일로 준비.\n"
                "1. Data Cleaning             : Data upload 후 Missing data 제거, 분류형 데이터 숫자로 변경후 저장.\n" "꼭 저장한 파일을 다시 Upload 해야 함 !!\n"
                "2. X, Y Description          : X,Y인자에 대한 Visulization 및 Correlation ratio & Heatmap을 통한 관계 확인.\n"
                "3. Feature Selection         : X-Y Performance Graph 결과를 통한 최종 X 인자 갯수 결정.\n"
                "4. Set Machine Learning Model: 각 모델별 결과를 비교하여 최적모델 선택.\n"
                "5. Model Optimization        : 선택된 모델 최적화 후 Train 파일과 모델 파일 저장.\n"
        )
    
#---------------------------------#
# Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Data Cleaning '):
    
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
       
    if st.sidebar.checkbox('Download Example1'):
        uploaded_file = './Example1.csv'
        
    if st.sidebar.checkbox('Download Example2'):
        uploaded_file = './DCB_test4.CSV'
        
    

#---------------------------------#
# Main panel

# Displays the dataset
    st.subheader('**1. Data Cleaning**')

    if uploaded_file is not None:
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        
        #if st.sidebar.checkbox('Download Example1'):
        #    uploaded_file = pd.read_csv('https://raw.githubusercontent.com/gigantic65/Test/main/Example1.csv')
            #st_pandas_to_csv_download_link2(File_save, file_name = "Example1.csv")
        
        df = load_csv()
    
        #st.session_state['df'] = df
    
        st.write('**1.1 Data set**')
        st.write(df)
        #st.dataframe(df,1500,500)
        
        st.write('**1.2 Data statistics**')
        st.write(df.describe())
    

    
        
        st.write('**1.3 Data Cleaning**')
        st.write('- Check Classification, Duplicate, Missing, Outlier data')
        
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        
        
        col1,col2 = st.columns([1,25])
        plt.style.use('classic')
        #fig, ax = plt.subplots(figsize=(10,10))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        with col2:
            Out = st.number_input('Outlier Criteria - Number of Sigma(σ)',0,10,3, format="%d")
        #    option = st.radio('Missing data/Outlier Managing Method?',('Remove', 'Fill'))
        #    if option == 'Fill':
        #        option = st.radio('Filling Method?',('Front_value', 'Back_value','Interpolate_value','Mean_value'))
        #Out = st.radio('Outlier Criteria - Number of Sigma(σ)',('2', '3','4', '5','6'),4)
        

        # Outlier 확인
        new = df.copy()
        le = preprocessing.LabelEncoder()
        for column in new.columns:
            if new[column].dtypes == 'O':
                new[column] = le.fit_transform(new[column])
        
        for i in range(len(new.columns)):
            
            df_std = np.std(new.iloc[:,i])
            df_mean = np.mean(new.iloc[:,i])
            cut_off = df_std * Out
            upper_limit = df_mean + cut_off
            lower_limit = df_mean - cut_off
            #print(upper_limit,df_mean,lower_limit)
            for j in range(new.shape[0]):
                if new.iloc[j,i] > upper_limit or new.iloc[j,i] <lower_limit:
                    new.iloc[j,i] = np.nan


        #결측치 % 확인
        Missing = pd.DataFrame()
        Missing['column'] = df.columns
        Missing['Classification_Type'] = 0
        
        for column in df.columns:
            if df[column].dtypes == 'O':
                Missing.Classification_Type.loc[Missing['column']==column] = 1

        Missing.set_index('column', inplace=True)
        #Missing['Classification'] = test['column2']
        Missing['Missing Data'] = df.isna().sum()
        Missing['Outlier'] = new.isna().sum()
        Missing['Unique Value'] = df.nunique()        
        

        
        #Outlier Check
        
        newlist =  ['<select>']
        
        
        for column in df.columns:
            if df[column].dtypes != 'O':
                newlist.append(column)
                
        
       # for column in df.columns:
       #     newlist.append(column)
        
        #newlist = newlist.append(newlist2)
        
        with col2:
            outlier2 = st.selectbox("Data Profile : ", newlist)
        
               
        for column in df.columns:
            if column == outlier2:
                asd= df[column].copy()
                
                asd = pd.DataFrame(asd)
                x = range(asd.shape[0])
                                
                asd['max'] = np.mean(asd)[0] + Out * np.std(asd)[0]  #Out = Outlier Criteria (σ)
                asd['min'] = np.mean(asd)[0] - Out * np.std(asd)[0]
                asd['mean'] = np.mean(asd)[0]
                
                sns.set(rc={'figure.figsize':(15,6)})
               
                plt.plot(x, asd, color = "black")
                
                plt.plot(x, asd['max'], color = "red", label='Outlier limit')
                plt.plot(x, asd['min'], color = "red")
                plt.plot(x, asd['mean'], color = "blue",label='Mean Value')
                plt.legend(loc='upper left', fontsize=15)
                
                st.pyplot()
        
        st.write(Missing)
        
        st.write('')
        
        
        st.write('- Manage Classification data')
        
        if st.button('Change Classification --> Numerical'):
            #df2 = df
            #df2 = pd.read_csv('my_file.csv')
            le = preprocessing.LabelEncoder()
            for column in df.columns:
                if df[column].dtypes == 'O':
                    df[column] = le.fit_transform(df[column])
            df2 = df
            st.write(df2)
            st_pandas_to_csv_download_link(df2, file_name = "Cleaned_classification_data.csv")
            st.write('*Save directory setting : right mouse button -> save link as') 
            #st.write('*After data saving, must reopen it !!')   
        
        st.write('- Manage Duplicate, Missing, Outlier data')
        
        
              
        #with col2:
        #        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        #        option = st.radio('Missing data/Outlier Managing Method?',('Remove', 'Fill'))
        #        if option == 'Fill':
        #            option = st.radio('Filling Method?',('Front_value', 'Back_value','Interpolate_value','Mean_value'))
                
        #st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        #option = st.radio('Missing data/Outlier Managing Method?',('Remove', 'Fill'))
        
        #if option == 'Fill':
        #    option = st.radio('Filling Method?',('Front_value', 'Back_value','Interpolate_value','Mean_value'))
            
       
        #with col2:
            #Out = st.number_input('Outlier Criteria - Number of Sigma(σ)',0,10,3, format="%d")
            
        col3,col4 = st.columns([1,25])
        
        plt.style.use('classic')
        #fig, ax = plt.subplots(figsize=(10,10))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        with col4:
            option = st.radio('Missing data/Outlier Managing Method?',('No_Change','Remove', 'Fill'))
            if option == 'Fill':
                option = st.radio('Filling Method?',('Zero_value','Front_value', 'Back_value','Interpolate_value','Mean_value'))
        #Out = st.radio('Outlier Criteria - Number of Sigma(σ)',('2', '3','4', '5','6'),4)    
            
            
        if st.button('Change Duplicate, Missing, Outlier data'):
            
            # delete duplicated rows
            dupli = df.duplicated().sum()
            df = df.drop_duplicates()
            st.write('The number of Duplicate data :', dupli)
            
            #st.write('after dupilicate :', df.shape[0])
            
            # delete missing value   
            miss = df.isnull().sum().sum()
            #df = df.dropna()
            st.write('The number of Missing data :',miss)
            
            
            #st.write('after missing :', df.shape[0])
            # detect outlier
            
            le = preprocessing.LabelEncoder()
            for column in df.columns:
                if df[column].dtypes == 'O':
                    df[column] = le.fit_transform(df[column])
            
            for i in range(len(df.columns)):
    
                df_std = np.std(df.iloc[:,i])
                df_mean = np.mean(df.iloc[:,i])
                cut_off = df_std * Out
                upper_limit = df_mean + cut_off
                lower_limit = df_mean - cut_off
                
                for j in range(df.shape[0]):
                    if df.iloc[j,i] > upper_limit or df.iloc[j,i] <lower_limit:
                        df.iloc[j,i] = np.nan
        
            st.write('The number of Outlier data :', df.isna().sum().sum())
            
            #d_line0 = df.shape[0]
            #df = df.dropna()
            #d_line1 = df.shape[0]
         
            #st.write('The number of Outlier data :', d_line0-d_line1)    
            #st.write(df)
            if option == 'No_Change':
                df2 = df
            
            elif option == 'Remove':
                # delete missing value    
                df2 = df.dropna().reset_index()
                df2.drop(['index'],axis=1,inplace=True)
                
            elif option == 'Zero_value':
                df2 = df.fillna(0)
                
            elif option == 'Front_value':
                df2 = df.fillna(method='pad')
            
            elif option == 'Back_value':
                df2 = df.fillna(method='bfill')
                
            elif option == 'Interpolate_value':
                df2 = df.fillna(df.interpolate())
                
            elif option == 'Mean_value':
                df2 = df.fillna(df.mean())
                

          
            #st.write('after outlier :', df2.shape[0])
            st.write(df2)

            st_pandas_to_csv_download_link(df2, file_name = "Final_cleaned_data.csv")
            
            st.write('*Save directory setting : right mouse button -> save link as')
            
            df2 = pd.DataFrame(df2)
            
            st.session_state['df2'] = df2
            
            st.write(st.session_state['df'])
            st.write(st.session_state['df2'])

        else:
            
            le = preprocessing.LabelEncoder()
            for column in df.columns:
                if df[column].dtypes == 'O':
                    df[column] = le.fit_transform(df[column])
                    
            
            for i in range(len(df.columns)):
    
                df_std = np.std(df.iloc[:,i])
                df_mean = np.mean(df.iloc[:,i])
                cut_off = df_std * Out
                upper_limit = df_mean + cut_off
                lower_limit = df_mean - cut_off
                
                for j in range(df.shape[0]):
                    if df.iloc[j,i] > upper_limit or df.iloc[j,i] <lower_limit:
                        df.iloc[j,i] = np.nan
                        
            df2 = df.dropna().reset_index()
            df2.drop(['index'],axis=1,inplace=True)

            #st.write('after outlier :', df2.shape[0])

            
            df2 = pd.DataFrame(df2)
            
            #df2 = st.session_state['df2']
            
            #st.write('check')
            #st.write(df2)
        
        #df2 = pd.read_csv('df2.csv')
#    st.write('---')
#    st.header('**Pandas Profiling Report**')
        
        st.subheader('**2. X, Y Description**')
        with st.sidebar.header('2. X, Y Description'):
            
            #df2 = st.session_state['df2']
            
        
            x = list(df2.columns)
            #y = list(df2.columns[df2.shape[1]-2:])

            Selected_X = st.sidebar.multiselect('X variables', x, x)
            
            y = [a for a in x if a not in Selected_X]
            
            Selected_y = st.sidebar.multiselect('Y variables', y, y)
        
            Selected_X = np.array(Selected_X)
            Selected_y = np.array(Selected_y)
            
        if Selected_y.shape[0] <= 1:
                   
            st.write('**2.1 Number of X Variables:**',Selected_X.shape[0])
            st.info(list(Selected_X))
        
            st.write('**2.2 Number of Y Variables:**',Selected_y.shape[0])
            st.info(list(Selected_y))
    
            df3 = pd.concat([df2[Selected_X],df2[Selected_y]], axis=1)
            #df2 = df[df.columns == Selected_X]
    
            st.write(df3)
        
            #Selected_xy = np.array((Selected_X,Selected_y))
        
            st.write('**2.3 X, Y Visualization**')
            vis_col = ['All_variables']
            vis_col = pd.DataFrame(vis_col)

            test = list(df3.columns)
            test = pd.DataFrame(test)

            vis_col2 = vis_col.append(test)

            visual = st.multiselect('Choose Parameter for Visualization',vis_col2)
            if visual == ['All_variables']:
                visual = df3.columns
            
        #st.write(visual)
        #st.write(df3.columns)
            if st.button('Data Visualization'):
                col1,col2 = st.columns([1,1])
                plt.style.use('classic')
                #fig, ax = plt.subplots(figsize=(10,10))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                with col1:
                        #st.write('**X variable distribution**')
                        st.markdown("<h6 style='text-align: center; color: black;'>X variable distribution</h6>", unsafe_allow_html=True)
                with col2:
                        st.markdown("<h6 style='text-align: center; color: black;'>X - Y Graph</h6>", unsafe_allow_html=True)
            
                sns.set(font_scale = 0.8,rc={'figure.figsize':(10,5)})
            #plt.figure(figsize=(5,10))
                            
                for vis in visual:
                
                    fig, axs = plt.subplots(ncols=2)
                
                    g = sns.distplot(df3[vis], hist_kws={'alpha':0.5}, bins=8, kde_kws={'color':'xkcd:purple','lw':3}, ax=axs[0])

                    g2 = sns.scatterplot(x=df3[vis],y=df3.iloc[:,-1],s=60,color='red', ax=axs[1])

                    g2 = sns.regplot(x=df3[vis],y=df3.iloc[:,-1], scatter=False, ax=axs[1])

                    st.pyplot()
                
        

        
            if st.button('Intercorrelation X,Y Matrix'):
            
                df_cor = df3.corr()
                st.write(df_cor)
        
                df3.to_csv('output.csv',index=False)
                df3 = pd.read_csv('output.csv')

                corr = df3.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                sns.set(rc = {'figure.figsize':(6,6)},font_scale=0.5)
                #sns.set(font_scale=0.4)
                with sns.axes_style("white"):
                    f, ax = plt.subplots()
                    #ax = sns.heatmap(corr,mask=mask, vmax=1.0, square=True, annot = True, cbar_kws={"shrink": 0.7}, cmap='coolwarm', linewidths=.5)
                    ax = sns.heatmap(corr,  vmax=1, square=True, cbar_kws={"shrink": 0.7}, annot = True, cmap='coolwarm', linewidths=.5)
                st.set_option('deprecation.showPyplotGlobalUse', False)
        
                st.pyplot()
        

            st.subheader('**3. X Variables Selection**')
        
            st.write('**3.1 X variables Selection Method Results**')  
        
            #if st.button('Feature Selection Method (RFE)'):
        
            st.sidebar.header('3. X Variables Selection')
        
                   
            if st.sidebar.button('X preformance check(RFE Method)'):
                feature_s(df3)
        
            with st.sidebar.markdown('**Choose X variables**'):
            
                fs_list = []
                for i in range(1,df3.shape[1]):
                    fs_list.append(i)
        
                hobby = st.sidebar.selectbox("Optimum number of X variables : ", fs_list)
            
            
                X_rfe_columns = F_feature(df3,hobby)
            
        
                X_column = pd.DataFrame(X_rfe_columns,columns=['Variables'])
        
            #st.write(X_column)
            
            #y_column = list(df3.columns[df.shape[1]-1:])
            #st.write(X_column)
            
                count=0
            #list_x = list(X_column.Variables[0:hobby])
                Selected_X2 = list(X_column.Variables)
            #list_y = list(Selected_y)
                Selected_y = list(Selected_y)
    
            #Selected_X2 = st.sidebar.multiselect('X variables', list_x, list_x, key = count)
            #st.sidebar.multiselect('Y variables', list_y, list_y, key = count)
                count +=1
            

            #st.info(list(X_rfe_columns))
            


       
        
            df3 = pd.concat([df2[Selected_X2],df2[Selected_y]], axis=1)
        
            st.write('**3.2 The Selected Final Variables & Data**')
    
        #st.write('**3.1 Selected X, Y Variables**')
            st.write('**Number of Final X Variables:**',len(Selected_X2))
            st.info(list(Selected_X2))
    
            st.write('**Number of Final Y Variables:**',len(Selected_y))
            st.info(list(Selected_y))
    
            #st.write(df3)    
        #st.write(X_column)
        #st.write(list_x)
    

    #st.write(df3)
    



            st.subheader('**4. Machine Learning Model Comparison**')
        #sns.set(rc={'figure.figsize':(10,10)})
        #correlation = sns.heatmap(df.corr(), annot = False, cmap='coolwarm', linewidths=.5)
        #bottom, top = correlation.get_ylim()
        #correlation.set_ylim(bottom+0.5, top-0.5)
        #plt.show()
        
        
        #with st.sidebar.header('2. Feature Selection '):
            with st.sidebar.header('4. ML Model Comparision'):
        
                ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','AB','XGBOOST','Extra Trees','RandomForest']
                Selected_ml = st.sidebar.multiselect('Choosing ML Algorithm', ml, ml)
    
            if st.sidebar.button('Machine Learning Algorithm Comparison'):
                M_list = build_model(df3, Selected_ml)
                M_list = list(M_list['Machine_Learning_Model'][:3])
                M_list = pd.DataFrame(M_list)
                M_list.to_csv('test.csv')
            #st.write(test)
            
            else:
                st.markdown('**4.1. Model Validation Method:** K-Fold Cross Validation')
                st.markdown('**4.2. Machine Learning Model Comparison Results**')

            
            
    
          
        
            with st.sidebar.header('5. Model Optimization'):
            
                st.sidebar.markdown('**5.1. Voting Optimization**')
            
                ml = ['Linear Regression','Lasso','KNN','Decision_Tree','AB','GBM','XGBOOST','Extra Trees','RandomForest']
            
            #ml2 = M_list
            
                if os.path.isfile('test.csv'):
                    ml2 = pd.read_csv('test.csv')
                    #st.write(test)
                    ml2 = list(ml2.iloc[:,1].values)
                else:
                    ml2 = ['GBM','XGBOOST','Extra Trees','RandomForest']
            
            
                Selected_ml2 = st.sidebar.multiselect('Choosing Voting Algorithm', ml, ml2)
                
                Selected_ml3 = list(Selected_ml2)
            

        
        
            st.subheader('**5. Model Optimization**')
    
         
        
            st.markdown('**5.1. Voting Optimization Results**')
            
        
            
        
            if st.sidebar.button('Voting Optimization'):
            
            
                X = df3.iloc[:,:-1] # Using all column except for the last column as X
                Y = df3.iloc[:,-1] # Selecting the last column as Y
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            

                 
            

            #rescaledX = scaler.transform(X_train)


                models =[]
                
                for model in Selected_ml3:
                    
                
                    if model == 'Linear Regression':
                        models.append(('Linear Regression', Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])))
                    if model == 'Lasso':
                        models.append(('LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
                    if model == 'KNN':
                        models.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))      
                    if model == 'Decision_Tree':
                        models.append(('Decision_Tree', Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())])))   
                    if model == 'GBM':
                        models.append(('GBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor(n_estimators=75))])))
                    if model == 'XGBOOST':
                        models.append(('XGBOOST', Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(booster='gbtree',n_estimators= 100))])))
                    if model == 'AB':
                        models.append(('AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
                    if model == 'Extra Trees':
                        models.append(('Extra Trees', Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor())])))
                    if model == 'RandomForest':
                        models.append(('RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor())])))
            

                k=0
                
                         
                
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
            #    names.append(name)
                msg.append('%s' % model)
                mean.append('%f' %  (cv_results.mean()))
                std.append('%f' % (cv_results.std()))
                    
                    
                F_result3 = pd.DataFrame(np.transpose(msg))
                F_result3.columns = ['Machine_Learning_Model']
                F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
            
            #st.write(F_result3)    
        
        
                st.write('**Optimum Voting Model Performance**')
            
                st.write('Voting Model Accuracy ($R^2$):')
            
                R2_mean = list(F_result3['R2_Mean'].values)
                st.info( R2_mean[0] )
                
                st.write('Model Accuracy Deviation (Standard Deviation):')
            
                R2_std = list(F_result3['R2_Std'].values)
                st.info( R2_std[0])
            
            
                
            #scaler = StandardScaler().fit(X)
            #rescaled = scaler.transform(X)
            
            
                predictions = model.predict(X)
                predictions = pd.DataFrame(predictions).values              
        
        
            #st.markdown('*Voting Model Results*')
            #st.write(F_result3)
            
            
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.plot(Y, Y, color='blue', label = 'Actual data')
                plt.scatter(Y, predictions, color='red', label = 'Prediction')
                plt.xlabel(df3.columns[-1])
                plt.ylabel(df3.columns[-1])
                #plt.ylim(39.7,39.9) 
                #plt.xlim(39.7, 39.9) 
                plt.legend()
                st.pyplot()
            
        
        
                st.markdown('**Download Train file & Model file for Prediction**')
                
                st_pandas_to_csv_download_link(df3, file_name = "Train_File.csv")
        
                download_model(k,model)

                st.write('*파일저장: 왼쪽 마우스키 --> Download Folder에 저장, 오른쪽 마우스키 --> 링크저장 --> 원하는 위치에 저장') 
                



            with st.sidebar.markdown('**5.2. Hyper Parameter Optimization**'):
            
                max_num = df3.shape[1]
                ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','XGBOOST','Extra Trees','RandomForest','Neural Network']
                Model = st.sidebar.selectbox('Hyper Parameter Tuning',Selected_ml)
                #if Model == 'Linear Regression'or'Lasso':
                    #    parameter_n_neighbers = st.sidebar.slider('Number of neighbers', 2, 10, 6, 2)
                    # st.sidebar.markdown('No Hyper Parameter)
                if Model == 'KNN':
                    parameter_n_neighbors_knn = st.sidebar.slider('Number of neighbers', 2, 10, (2,8), 2)
                    parameter_n_neighbors_step_knn = st.sidebar.number_input('Step size for n_neighbors', 1)
                    n_neighbors_range = np.arange(parameter_n_neighbors_knn[0], parameter_n_neighbors_knn[1]+parameter_n_neighbors_step_knn, parameter_n_neighbors_step_knn)
                    param_grid_knn = dict(n_neighbors=n_neighbors_range)
            
                elif Model == 'GBM' or Model == 'Extra Trees' or Model == 'RandomForest':
                    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 301, (11,151), 20)
                    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                    parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, max_num, (1,3), 1)
                    parameter_max_features_step = st.sidebar.number_input('Step size for max_features', 1)
                    #parameter_max_depth = st.sidebar.slider('Number of max_depth (max_depth)', 10, 100, (30,80), 10)
                    #parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 10)
                    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+parameter_max_features_step, parameter_max_features_step)
                    #max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
                
                elif Model == 'AB' :
                    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 301, (11,151), 20)
                    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                    parameter_learning_rate = st.sidebar.slider('learning_rate', 0.1, 2.0, (0.1,0.6), 0.2)
                    parameter_learning_rate_step = st.sidebar.number_input('Step size for learing_rate', 0.2)
                    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                    learning_rate_range = np.arange(parameter_learning_rate[0], parameter_learning_rate[1]+parameter_learning_rate_step, parameter_learning_rate_step)
                    param_grid = dict(learning_rate=learning_rate_range, n_estimators=n_estimators_range)
                
            
                elif Model == 'XGBOOST' :
                    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 301, (41,101), 20)
                    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                    parameter_max_depth = st.sidebar.slider('max_depth', 0, 10, (2,5), 1)
                    parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 1)
                    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                    max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                    param_grid = dict(max_depth=max_depth_range, n_estimators=n_estimators_range)
            
                elif Model == 'Neural Network':
                    parameter_n_estimators = st.sidebar.slider('Number of first nodes', 10, 100, (10,40), 10)
                    parameter_n_estimators_step = st.sidebar.number_input('Step size for first nodes', 10)
                    parameter_n_estimators2 = st.sidebar.slider('Number of Second nodes', 10, 100, (10,40), 10)
                    parameter_n_estimators_step2 = st.sidebar.number_input('Step size for second nodes', 10)
                    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                    n_estimators_range2 = np.arange(parameter_n_estimators2[0], parameter_n_estimators2[1]+parameter_n_estimators_step2, parameter_n_estimators_step2)
                    param_grid = dict(n_estimators=n_estimators_range, n_estimators2=n_estimators_range2)
            
                elif Model == 'Linear Regression' or Model == 'Lasso' or Model == 'Decision_Tree':
                    st.sidebar.write(' No hyper parameter tuning')




            st.markdown('**5.2. Hyperparameter Optimization Results**')        
    
            if st.sidebar.button('Model Optimization'):
        
                X = df3.iloc[:,:-1] # Using all column except for the last column as X
                Y = df3.iloc[:,-1] # Selecting the last column as Y

                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
                #rescaled = scaler.transform(X)
                #rescaledX = scaler.transform(X_train)
                #rescaledTestX = scaler.transform(X_test)
            
                k=0
                if Model == 'Linear Regression':
                    #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])
                    #Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])))
                
        
                elif Model == 'Lasso':
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('Lasso',Lasso())])
                #model.fit(rescaled, y_train)

                
                elif Model == 'Decision_Tree':
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())])
                #model.fit(rescaled, y_train)
                
                            
                elif Model == 'KNN':
            
                    a = Opti_KNN_model(df3,parameter_n_neighbors_knn,parameter_n_neighbors_step_knn,param_grid_knn)
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor(n_neighbors=a))])
                #model.fit(rescaledX, y_train)
        
                elif Model == 'GBM':
                    a, b = Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid)
                #st.write(a, b)
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor(n_estimators=a, max_features=b))])
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
                
                elif Model == 'AB':
                    a, b = Opti_model3(Model,df3,parameter_n_estimators,parameter_learning_rate,param_grid)
                #st.write(a, b)
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor(n_estimators=a, learning_rate=b))])
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
            
                elif Model == 'XGBOOST':
                    a, b = Opti_model2(Model,df3,parameter_n_estimators,parameter_max_depth,param_grid)
                #st.write(a, b)
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(booster='gbtree',n_estimators=a, max_depth=b))])
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
                
                elif Model == 'Extra Trees':
                    a, b = Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid)
                #st.write(a, b)
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor(n_estimators=a, max_features=b))])
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
                elif Model == 'RandomForest':
                    a, b = Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid)
                #st.write(a, b)
                #    print(X_train, y_train)
                    model = Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor(n_estimators=a, max_features=b))])
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)


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
                #    names.append(name)
                msg.append('%s' % Model)
                mean.append('%f' %  (cv_results.mean()))
                std.append('%f' % (cv_results.std()))
                    
                    
                F_result3 = pd.DataFrame(np.transpose(msg))
                F_result3.columns = ['Machine_Learning_Model']
                F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
            
                #st.write(F_result3)    
        
        
                st.write('**Optimum Model Performance**')
                
                st.write('Final Model Accuracy ($R^2$):')
            
                R2_mean = list(F_result3['R2_Mean'].values)
                st.info( R2_mean[0] )
                
                st.write('Model Accuracy Deviation (Standard Deviation):')
            
                R2_std = list(F_result3['R2_Std'].values)
                st.info( R2_std[0])
                
        
                model.fit(X_train,y_train)
            
                predictions = model.predict(X)
                predictions = pd.DataFrame(predictions).values
 
            
            
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.plot(Y, Y, color='blue', label = 'Actual data')
                plt.scatter(Y, predictions, color='red', label = 'Prediction')
                plt.xlabel(df3.columns[-1])
                plt.ylabel(df3.columns[-1])
                #plt.ylim(39.7,39.9) 
                #plt.xlim(39.7, 39.9) 
                plt.legend()
                st.pyplot()
            
        
                st.markdown('**Download Train file & Model file for Prediction**')
        
                st_pandas_to_csv_download_link(df3, file_name = "Train_File.csv")
        
                download_model(k,model)
        
                st.write('*파일저장: 왼쪽 마우스키 --> Download Folder에 저장, 오른쪽 마우스키 --> 링크저장 --> 원하는 위치에 저장') 



        else :

  
            st.write('**2.1 Number of X Variables:**',Selected_X.shape[0])
            st.info(list(Selected_X))
        
            st.write('**2.2 Number of Y Variables:**',Selected_y.shape[0])
            st.info(list(Selected_y))
    
            df3 = pd.concat([df2[Selected_X],df2[Selected_y]], axis=1)
            #df2 = df[df.columns == Selected_X]
    
            st.write(df3)
        
            #Selected_xy = np.array((Selected_X,Selected_y))
        
            st.write('**2.3 X, Y Visualization**')
            vis_col = ['All_variables']
            vis_col = pd.DataFrame(vis_col)

            test = list(df3.columns)
            test = pd.DataFrame(test)

            vis_col2 = vis_col.append(test)

            visual = st.multiselect('Choose Parameter for Visualization',vis_col2)
            if visual == ['All_variables']:
                visual = df3.columns
            
 
            if st.button('Data Visualization', key = 1):
                col1,col2 = st.columns([1,1])
                plt.style.use('classic')
                #fig, ax = plt.subplots(figsize=(10,10))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                with col1:
                        #st.write('**X variable distribution**')
                        st.markdown("<h6 style='text-align: center; color: black;'>X variable distribution</h6>", unsafe_allow_html=True)
                with col2:
                        st.markdown("<h6 style='text-align: center; color: black;'>X - Y Graph</h6>", unsafe_allow_html=True)
            
                sns.set(font_scale = 0.8,rc={'figure.figsize':(10,5)})
            #plt.figure(figsize=(5,10))
                            
                for vis in visual:
                
                    fig, axs = plt.subplots(ncols=2)
                
                    g = sns.distplot(df3[vis], hist_kws={'alpha':0.5}, bins=8, kde_kws={'color':'xkcd:purple','lw':3}, ax=axs[0])
                    color = ['red','blue','green']
                    i=0
                    for feature in Selected_y:
                    #g2 = sns.scatterplot(x=df3[vis],y=df3.iloc[:,-1],s=60,color='red', ax=axs[1])
                        g2 = sns.scatterplot(x=df3[vis],y=df3[feature],s=60,color=color[i],ax=axs[1],label=feature)
                    #g2 = sns.regplot(x=df3[vis],y=df3.iloc[:,-1], scatter=False, ax=axs[1])
                        i+=1
                
                    st.pyplot()
                
        

        
            if st.button('Intercorrelation X,Y Matrix'):
            
                df_cor = df3.corr()
                st.write(df_cor)
        
                df3.to_csv('output.csv',index=False)
                df3 = pd.read_csv('output.csv')

                corr = df3.corr()
                mask = np.zeros_like(corr)
                mask[np.triu_indices_from(mask)] = True
                sns.set(rc = {'figure.figsize':(6,6)},font_scale=0.5)
                #sns.set(font_scale=0.4)
                with sns.axes_style("white"):
                    f, ax = plt.subplots()
                    #ax = sns.heatmap(corr,mask=mask, vmax=1.0, square=True, annot = True, cbar_kws={"shrink": 0.7}, cmap='coolwarm', linewidths=.5)
                    ax = sns.heatmap(corr,  vmax=1, square=True, cbar_kws={"shrink": 0.7}, annot = True, cmap='coolwarm', linewidths=.5)
                st.set_option('deprecation.showPyplotGlobalUse', False)
        
                st.pyplot()
        

            st.subheader('**3. X Variables Selection**')
        
            st.write('**3.1 X variables Selection Method Results**')  
        
           

            st.sidebar.header('3. X Variables Selection')
        
            
            if st.sidebar.button('X preformance check'):
                
                feature_m(df3,Selected_X,Selected_y )
                
                
        
            with st.sidebar.markdown('**Choose X variables**'):
            
        
                fs_list2 = []
                
                for i in range(1,df3.shape[1]):
                    fs_list2.append(i)
                    
        
                hobby2 = st.sidebar.selectbox("Optimum number of X variables : ", fs_list2)
                
                
                X_rfe_columns = F_feature_m(df3,hobby2,Selected_X,Selected_y)
            
        
                X_column = pd.DataFrame(X_rfe_columns,columns=['Variables'])
                
                
                
        
            #st.write(X_column)
            
            #y_column = list(df3.columns[df.shape[1]-1:])
            #st.write(X_column)
            
                count=0
            #list_x = list(X_column.Variables[0:hobby])
                Selected_X2 = list(X_column.Variables)
            #list_y = list(Selected_y)
                Selected_y = list(Selected_y)
    
            #Selected_X2 = st.sidebar.multiselect('X variables', list_x, list_x, key = count)
            #st.sidebar.multiselect('Y variables', list_y, list_y, key = count)
                count +=1
            

            #st.info(list(X_rfe_columns))
            


       
        
            df3 = pd.concat([df2[Selected_X2],df2[Selected_y]], axis=1)
        
            st.write('**3.2 The Selected Final Variables & Data**')
    
        #st.write('**3.1 Selected X, Y Variables**')
            st.write('**Number of Final X Variables:**',len(Selected_X2))
            st.info(list(Selected_X2))
    
            st.write('**Number of Final Y Variables:**',len(Selected_y))
            st.info(list(Selected_y))
    



            st.subheader('**4. Machine Learning Model Comparison**')
        #sns.set(rc={'figure.figsize':(10,10)})
        #correlation = sns.heatmap(df.corr(), annot = False, cmap='coolwarm', linewidths=.5)
        #bottom, top = correlation.get_ylim()
        #correlation.set_ylim(bottom+0.5, top-0.5)
        #plt.show()
        
        
        #with st.sidebar.header('2. Feature Selection '):
            with st.sidebar.header('4. ML Model Comparision'):
        
                ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','AB','XGBOOST','Extra Trees','RandomForest']
                Selected_ml = st.sidebar.multiselect('Choosing ML Algorithm', ml, ml)
    
            if st.sidebar.button('Machine Learning Algorithm Comparison'):
                M_list = build_model_m(df3, Selected_ml, Selected_X2,Selected_y)
                #M_list = list(M_list['Machine_Learning_Model'][:3])
                #M_list = pd.DataFrame(M_list)
                #M_list.to_csv('test.csv')
            #st.write(test)
            
            else:
                st.markdown('**4.1. Model Validation Method:** K-Fold Cross Validation')
                st.markdown('**4.2. Machine Learning Model Comparison Results**')

            
            
            st.sidebar.header('5. Model Optimization')
        
            st.subheader('**5. Model Optimization**')
    

            with st.sidebar.markdown('**5.1. Hyper Parameter Optimization**'):
            
                max_num = df3.shape[1]
                ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','XGBOOST','Extra Trees','RandomForest','Neural Network']
                Model = st.sidebar.selectbox('Hyper Parameter Tuning',Selected_ml)
                #if Model == 'Linear Regression'or'Lasso':
                    #    parameter_n_neighbers = st.sidebar.slider('Number of neighbers', 2, 10, 6, 2)
                    # st.sidebar.markdown('No Hyper Parameter)
                if Model == 'KNN':
                    parameter_n_neighbors_knn = st.sidebar.slider('Number of neighbers', 2, 10, (2,8), 2)
                    parameter_n_neighbors_step_knn = st.sidebar.number_input('Step size for n_neighbors', 1)
                    n_neighbors_range = np.arange(parameter_n_neighbors_knn[0], parameter_n_neighbors_knn[1]+parameter_n_neighbors_step_knn, parameter_n_neighbors_step_knn)
                    param_grid_knn = dict(estimator__n_neighbors=n_neighbors_range)
            
                elif Model == 'GBM' or Model == 'Extra Trees' or Model == 'RandomForest':
                    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 501, (101,251), 30)
                    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 30)
                    parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, max_num, (1,3), 1)
                    parameter_max_features_step = st.sidebar.number_input('Step size for max_features', 1)
                    #parameter_max_depth = st.sidebar.slider('Number of max_depth (max_depth)', 10, 100, (30,80), 10)
                    #parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 10)
                    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                    max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+parameter_max_features_step, parameter_max_features_step)
                    #max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                    param_grid = dict(estimator__max_features=max_features_range, estimator__n_estimators=n_estimators_range)
                
                elif Model == 'AB' :
                    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 501, (101,251), 20)
                    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 30)
                    parameter_learning_rate = st.sidebar.slider('learning_rate', 0.1, 2.0, (0.1,0.6), 0.2)
                    parameter_learning_rate_step = st.sidebar.number_input('Step size for learing_rate', 0.2)
                    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                    learning_rate_range = np.arange(parameter_learning_rate[0], parameter_learning_rate[1]+parameter_learning_rate_step, parameter_learning_rate_step)
                    param_grid = dict(estimator__learning_rate=learning_rate_range, estimator__n_estimators=n_estimators_range)
                
            
                elif Model == 'XGBOOST' :
                    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 301, (41,101), 20)
                    parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                    parameter_max_depth = st.sidebar.slider('max_depth', 0, 10, (2,5), 1)
                    parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 1)
                    n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                    max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                    param_grid = dict(estimator__max_depth=max_depth_range, estimator__n_estimators=n_estimators_range)
            

                elif Model == 'Linear Regression' or Model == 'Lasso' or Model == 'Decision_Tree':
                    st.sidebar.write(' No hyper parameter tuning')





            st.markdown('**5.1. Hyperparameter Optimization Results**')        
    
            if st.sidebar.button('Model Optimization'):
                
                
                X = df3[Selected_X2] # Using all column except for the last column as X
                Y = df3[Selected_y] # Selecting the last column as Y
                
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        

            
                if Model == 'Linear Regression':
                    #    print(X_train, y_train)
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())]))

        
                elif Model == 'Lasso':
                #    print(X_train, y_train)
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())]))
                #model.fit(rescaled, y_train)

                
                elif Model == 'Decision_Tree':
                #    print(X_train, y_train)
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())]))
                #model.fit(rescaled, y_train)
                
                            
                elif Model == 'KNN':
                    
                    a = Opti_KNN_model_m(df3,param_grid_knn,Selected_X2,Selected_y)
            
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor(n_neighbors=a))]))
                #model.fit(rescaledX, y_train)
        
                elif Model == 'GBM':
                    
                    a, b = Opti_model_m(Model,df3,param_grid,Selected_X2,Selected_y)
                    
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('GBM',GradientBoostingRegressor(n_estimators=a, max_features=b))]))
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
                
                elif Model == 'AB':
                    
                    a, b = Opti_model3_m(Model,df3,param_grid,Selected_X2,Selected_y)
                    
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor(n_estimators=a, learning_rate=b))]))
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
            
                elif Model == 'XGBOOST':
                    
                    a, b = Opti_model2_m(Model,df3,param_grid,Selected_X2,Selected_y)
             
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(n_estimators=a, max_depth=b))]))
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
                
                elif Model == 'Extra Trees':
                    
                    a, b = Opti_model_m(Model,df3,param_grid,Selected_X2,Selected_y)
                   
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor(n_estimators=a, max_features=b))]))
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)
                elif Model == 'RandomForest':
                    
                    a, b = Opti_model_m(Model,df3,param_grid,Selected_X2,Selected_y)
                    
                    model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor(n_estimators=a, max_features=b))]))
                #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                #model.fit(rescaledX, y_train)


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
                #    names.append(name)
                msg.append('%s' % Model)
                mean.append('%f' %  (cv_results.mean()))
                std.append('%f' % (cv_results.std()))
                    
                    
                F_result3 = pd.DataFrame(np.transpose(msg))
                F_result3.columns = ['Machine_Learning_Model']
                F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
            
                #st.write(F_result3)    
        
        
                st.write('**Optimum Model Performance**')
                
                st.write('Final Model Accuracy ($R^2$):')
            
                R2_mean = list(F_result3['R2_Mean'].values)
                st.info( R2_mean[0] )
                
                st.write('Model Accuracy Deviation (Standard Deviation):')
            
                R2_std = list(F_result3['R2_Std'].values)
                st.info( R2_std[0])
                
        
                model.fit(X_train,y_train)
            
                predictions = model.predict(X)
                predictions = pd.DataFrame(predictions)
 

                
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.figure(figsize=(10,6))
                fig, axs = plt.subplots(ncols=Y.shape[1])
                fig.subplots_adjust(hspace=1)
                
                


                for i in range(1,Y.shape[1]+1):
                

                    
                    plt.subplot(1,Y.shape[1],i)
                    
                    plt.plot(Y.iloc[:,i-1], Y.iloc[:,i-1], color='blue', label = 'Actual data')
                    plt.scatter(Y.iloc[:,i-1], predictions.iloc[:,i-1], color='red', label = 'Prediction')
                    plt.title(Y.columns[i-1],fontsize=10)
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=8)
                    #ax.set_xlabel('Time', fontsize=16)
                    #plt.ylabel(Y.columns[i-1], fontsize=10)
                    
                st.pyplot()
                    
                
                k=0
        
                st.markdown('**Download Train file & Model file for Prediction**')
        
                st_pandas_to_csv_download_link(df3, file_name = "Train_File.csv")
        
                download_model(k,model)
        
                st.write('*파일저장: 왼쪽 마우스키 --> Download Folder에 저장, 오른쪽 마우스키 --> 링크저장 --> 원하는 위치에 저장') 