import streamlit as st
from dotenv import load_dotenv
import os 
from utils import *

def main():

    st.set_page_config(page_title="PDF Summarizer")
    st.write("Summarize your PDF here!")
    st.divider()

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    submit = st.button("Generate Summary")

    os.environ["api_key"] = os.getenv("OPENAI_API_KEY")

    if submit:
        response = summarizer(pdf)
        st.subheader("Summary of the File:")
        st.write(response)

if __name__ == '__main__':
    main()