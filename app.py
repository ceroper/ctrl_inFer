import streamlit as st
from annotated_text import annotated_text
from annotated_text import annotation
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame
import pandas as pd
import seaborn as sns
from functionforDownloadButtons import download_button
import json
import numpy as np
import string
import re

st.set_page_config(
    page_title="Ctrl+inFer",
    page_icon="đ§ ",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("Ctrl+inFer")
    st.header("")



with st.expander("âšī¸ - About this app", expanded=True):

    st.write(
        """     
        This tool identifies synonymns to a search term in the text. You can think of it as a smarter version of ctrl+f.
        """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **đ Paste document **")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        Threshold = st.slider(
            "Keyword similarity threshold",
            value=0.7,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""Higher values scan for more similar synonyms.""",
        )

    with c2:
        keyword = st.text_area(
            "Paste your text below (1 word)",
            height=50,
        )
        
        doc = st.text_area(
            "Paste your text below (max 500 words)",
            height=510,
        )
        
        MAX_KEYWORDS = 1
        res = len(keyword.split())
        if res > MAX_KEYWORDS:
            st.warning(
                "â ī¸ Your keyword contains "
                + str(res)
                + " words."
                + " Only the first word will be reviewed."
            )

            keyword = ' '.join(keyword.split()[:MAX_KEYWORDS])
        
        MAX_WORDS = 500
        res = len(doc.split())
        if res > MAX_WORDS:
            st.warning(
                "â ī¸ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed."
            )

            doc = ' '.join(doc.split()[:MAX_WORDS])

        submit_button = st.form_submit_button(label="Flag all the similar keywords")

if not submit_button:
    st.stop()

def space_before(x):
    if x[0] in string.punctuation.replace('(', '').replace('-', ''):
        return ''
    else:
        return ' '

def check_threshold(x, word, threshold):
    if cosine_similarity(x.vector.reshape(1, -1), word.reshape(1, -1)) > Threshold:
        # could change this to not always put a space between - don't want a space between a word and puncutation
        return ((space_before(str(x)) + str(x), ''))
    else:
        return (space_before(str(x)) + str(x))

def find_keywords(doc, keyword, Threshold):
    nlp = spacy.load('en_core_web_md')

    doc = nlp(doc)
    word = nlp(keyword).vector

    to_highlight = [str(x) for x in doc if cosine_similarity(x.vector.reshape(1, -1), word.reshape(1, -1)) > Threshold]

    return (to_highlight)

def highlight_keywords(doc, keyword, Threshold):
    nlp = spacy.load('en_core_web_md')
    
    doc = nlp(doc)
    word = nlp(keyword).vector
    
    highlighted = [check_threshold(x, word, Threshold) for x in doc]
    highlighted[0] = highlighted[0].lstrip()
    
    return (highlighted)

keywords = find_keywords(doc, keyword, Threshold)

highlighted = highlight_keywords(doc, keyword, Threshold)

annotated_text(*highlighted)

st.markdown("## ** Check & download results **")

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "đĨ Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "đĨ Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "đĨ Download (.json)")

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase"])
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
)

c1, c2, c3 = st.columns([1, 3, 1])

with c2:
    st.table(df)
