import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from htmlTemp import css, user_template
from rag.retrieving import reply, display_pdf, get_epub_text, get_pdf_text, get_url_text, get_chunks, get_vectorstore
from images import my_im, bot_im
import validators

import os


llm = Ollama(model="mistral")


def reset_chat():
    st.session_state.vectorstore = None
    st.session_state.conversation = None
    st.session_state.chat_history = []
    st.session_state.url_list = []
    st.rerun()


def is_valid_url(url):
    return validators.url(url)


def main():

    # Set page config and theme
    load_dotenv()
    st.set_page_config(
        page_title="Datachat",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded"  # Optional: Expand the sidebar by default
    )

    st.write(css, unsafe_allow_html=True)
    # App layout
    col1, col2 = st.columns([10, 1])

    with col1:
        st.header(f"Ask multiple websites , files and other documents :books:")

    with col2:
        clear = st.button("Clear ↺")
        if clear:
            reset_chat()

    # st.header("Chat with multiple PDF/epub")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "url_list" not in st.session_state:
        st.session_state.url_list = []

    if "process" not in st.session_state:
        st.session_state.process = False

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            if message["role"] == "user":
                st.write(message["html"].replace(
                    "{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(message["content"])

    user_input = st.chat_input("Ask the Document")
    if user_input:
        with st.chat_message("user", avatar=my_im):
            st.write(user_template.replace(
                "{{MSG}}", user_input), unsafe_allow_html=True)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "html": user_template, "avatar": my_im})
        if st.session_state.process:
            reply(user_input)
        else:
            st.error("Please click on :red[ **Process** ] after uploading ! 😉")

    col1, col2 = st.columns([6, 1])

    with st.sidebar:
        st.subheader("Put Your Websites Here")
        url = st.text_input("Website URls")
        if url:
            if is_valid_url(url):
                st.session_state.url_list.append(url)
                st.session_state.url_list = list(
                    set(st.session_state.url_list))
            else:
                # so it doesn't add (duplicate) the last element when pressing process and the input widget is not empty
                st.write(":red[Please enter a valid url]")
            for url in st.session_state.url_list:
                st.write(f"* {url} was **added**")

        st.subheader("Put Your Documents Here")
        files = st.file_uploader(
            "Upload",
            accept_multiple_files=True)
        if files is not None:
            with st.expander("See documents Preview"):
                for p in files:
                    if p.name.endswith(".pdf"):
                        display_pdf(p)

        if st.button("Process"):
            if len(st.session_state.url_list) == 0 and len(files) == 0:
                st.write(
                    ":red[Please enter a website or upload a documen first !]")
            else:
                st.session_state.process = True
                with st.spinner("Processing"):
                    # get the files/epubs text
                    txt = ""
                    if files is not None:
                        for p in files:
                            if p.name.endswith(".pdf"):
                                txt += get_pdf_text(p)
                            else:
                                txt += get_epub_text(p)
                    # get the websites text
                    url_txt = ""
                    if len(st.session_state.url_list) > 0:
                        docs = get_url_text(st.session_state.url_list)
                        for doc in docs:
                            url_txt += doc.page_content

                    # combine the text
                    raw = url_txt+txt

                    # chunking the text
                    chunks = get_chunks(raw)
                    # make a vectorstore
                    st.session_state.vectorstore = get_vectorstore(chunks)
                    st.info('Documents processed successfully', icon="🤖")


if __name__ == '__main__':
    main()
