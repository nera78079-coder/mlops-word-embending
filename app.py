import streamlit as st
import joblib
from word_embendding import transform_sentence_vector , transform_vector_padding
import numpy as np 



st.title("word embending")
raw_message = st.text_area("Write your text")

if st.button("Submit"):
    model = joblib.load('model.pkl')

    message_vector=transform_sentence_vector([raw_message])
    st.write("After convert eatch word to his unique identifier eatch line present phrase.")
    st.write(np.array(message_vector))

    message_padding=transform_vector_padding(message_vector)
    st.write("Add padding of zero to have same length or all phrases (padding=8) eatch line present phrase.")
    st.write(np.array(message_padding))

    st.write("Each line present embbending of each cell in padding table to 10 feauture.")
    st.write(model.predict(message_padding)[0])