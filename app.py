import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import home, Build_model_app, Prediction_app# import your pages here



# rest of the code

# Create an instance of the app 

app = MultiPage()

#st.set_page_config(layout="wide")
    
# Title of the main page
st.markdown("<h1 style='text-align: center; background-color:darkred; color: white;'>Machine Learning Application</h1>", unsafe_allow_html=True)

# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("Build Machine Learning Model", Build_model_app.app)
app.add_page("Predict New Conditions", Prediction_app.app)
#app.add_page("Monitoring", Monitor_app.app)


# The main app
app.run()