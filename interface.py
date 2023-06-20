import streamlit as st

from answer import *

# Terminal => streamlit run interface.py
st.set_page_config(
    page_title="Punctuation Corrector",
    page_icon="🟣",
    layout="centered",
)

st.title("Steblianko - Service for text correction using neural network")

left, right = st.columns(2)

if 'inb' not in st.session_state:
    st.session_state.inb = pd.Series(index=['x', 'y_mask', 'y_pred'], dtype=object, data=[[], [], []])


def update_res():
    st.session_state.res, st.session_state.inb = answer(st.session_state.inp)
    return


left.text_area("Text to be corrected", key="inp")
right.text_area("Correction result", key="res")
st.button("Submit", on_click=update_res)

if st.checkbox("Show inbetween steps"):
    one, two, three = st.columns(3)
    one.write("x")
    two.write("y_mask")
    three.write("y_pred")

    one.write(st.session_state.inb['x'])
    two.write(st.session_state.inb['y_mask'])
    three.write(st.session_state.inb['y_pred'])
