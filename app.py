from copyreg import pickle
import streamlit as st
import plotly.express as px
import numpy as np
import pickle

st.title("My Iris Predictor")
st.header("Let's predict Iris species")
st.subheader("Cool app, huh?")

df_iris = px.data.iris()
#df_iris

# hist_sl = px.histogram(df_iris, x='sepal_length')
# hist_sl

show_df = st.checkbox("Do you want to see the data?")
show_df

if show_df:
    df_iris

sl = st.number_input("Sepal Length (cm)", 0, 100)
sw = st.number_input("Sepal Width (cm)", 0, 100)
pl = st.number_input("Petal Length (cm)", 0, 100)
pw = st.number_input("Petal Width (cm)", 0, 100)

user_input = np.array([sl, sw, pl, pw]).reshape(1, -1)
# user_input

# Context block to open file and load
with open("saved-iris-model.pkl", "rb") as f:
    classifier = pickle.load(f)

prediction = classifier.predict(user_input)
# prediction

st.header(f"The model predicts: {prediction[0]}!")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")
    
with col2:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")
    
with col3:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")