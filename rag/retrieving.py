from langchain.chains import ConversationalRetrievalChain, create_history_aware_retriever, create_retrieval_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from images import my_im, bot_im
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import streamlit as st
import tempfile
import time
import textract
import base64
import torch
from sentence_transformers import SentenceTransformer
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder
from htmlTemp import css, bot_template, user_template
from langchain.output_parsers import StructuredOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
import os

os.environ["HF_HOME"] = "/teamspace/studios/this_studio/weights"
os.environ["TORCH_HOME"] = "/teamspace/studios/this_studio/weights"


persist_directory = None

llm = Ollama(model="mistral")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pdf_text(pdf):
    txt = ""
    pd = PdfReader(pdf)
    for page in pd.pages:
        txt += page.extract_text()
    return txt


def display_pdf(file):
    # Opening file from file path

    st.markdown(f"## {file.name}")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:50vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def get_epub_text(p):
    # Read the content of the uploaded file as bytes
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary file path within the temporary directory
          temp_file_path = os.path.join(temp_dir, p.name)

          # Save the uploaded file to the temporary path
           with open(temp_file_path, "wb") as f:
                f.write(p.getvalue())

            text = textract.process(temp_file_path).decode("utf-8")

            return text


def get_url_text(urllist):
    loader = WebBaseLoader(urllist)
    documents = loader.load()
    return documents


def get_chunks(raw):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size= 514,
        chunk_overlap=100,
        length_function= len
    )
    chunks = splitter.split_text(raw)
    return chunks


def get_vectorstore(chunks):
    model_name = 'WhereIsAI/UAE-Large-V1'
    model_kwargs = {'device': device}  # specify GPU device
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore


def get_context_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # f
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)
    return retriever_chain


def get_stuff_documents_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''Answer the user's questions based on the below context :\n\n{context}\n\n
      focus on the page content more than the metadata . if you don't find the answer to the user's question in the documents say the document does not provide 
      the requested information '''),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return stuff_documents_chain


def response_generator(res):
    lines = res.split("\n")  # Split text into lines
    for line in lines:
        words = line.split(" ")  # Split each line into words
        for word in words:
            yield word + " "
            time.sleep(0.02)
        yield "\n"  # Yield newline character after each line
        time.sleep(0.1)


def load_reranker_model(
    reranker_model_name: str = "BAAI/bge-reranker-large", device: str = "cuda"
) -> CrossEncoder:
    reranker_model = CrossEncoder(
        model_name=reranker_model_name, max_length=514, device=device
    )
    return reranker_model


def rerank_docs(reranker_model, query, retrieved_docs):
    query_and_docs = [(query, r.page_content) for r in retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    # we return the retrieved docs sorted based on their scores
    # and we only give the top 1/2 of the reranked docs
    return sorted(list(zip(retrieved_docs, scores)), key=lambda x: x[1], reverse=True)[:len(retrieved_docs)//2]
# d


def reply(question):
    # we get the retriever chain
    retriever_chain = get_context_retriever(st.session_state.vectorstore)
    # we Create a chain for passing a list of Documents to a model.
    conversation_rag_chain = get_stuff_documents_chain()

    # now we retrieve the documents that match our query and we rerank them with our reranker model
    # the reranked docs will then be passed in the context field.
    reranker_model = load_reranker_model()
    retrieved_docs = retriever_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": question
    })
    reranked_docs = rerank_docs(reranker_model, question, retrieved_docs)
    context = [inner_list[0] for inner_list in reranked_docs]

    # We get the response
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "context": context,
        "input": question})
    st.session_state.chat_history.append({"role": "ai", "content": response, "html": bot_template,"avatar":bot_im})
    with st.chat_message("ai", avatar=bot_im):
        st.write_stream(response_generator(response))
