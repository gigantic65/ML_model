
import streamlit as st



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
        

def app():
    
    st.markdown("<h5 style='text-align: right; color: black;'>KCC Ver. 1.0</h5>", unsafe_allow_html=True)
    
    st.write("")
    
    st.image('AI.png', width=None)
    
    st.write("")
    st.write("")
    st.write("")

    
    
    

    



#def remote_css(url):
#    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

#def icon(icon_name):
#    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

#local_css("style.css")
#remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


#---------------------------------#


# Timer
# Created by adapting from:
# https://www.geeksforgeeks.org/how-to-create-a-countdown-timer-using-python/
# https://docs.streamlit.io/en/latest/api.html#lay-out-your-app



    
 
