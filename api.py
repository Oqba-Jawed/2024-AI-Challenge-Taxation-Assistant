from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd
from PyPDF2 import PdfReader
import openai
import spacy
from semantic_split import SimilarSentenceSplitter, SentenceTransformersSimilarity, SpacySentenceSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import markdown
import os
import pickle
from flask_cors import CORS
import requests
import numpy as np

app = Flask(__name__)
CORS(app)

conversation = ""



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text

def get_text_chunks(raw_text):
    model = SentenceTransformersSimilarity()
    sentence_splitter = SpacySentenceSplitter()
    splitter = SimilarSentenceSplitter(model, sentence_splitter)
    chunks = splitter.split(raw_text)
    return chunks

def get_vectorstore(text_chunks, vectorstore_filename="vectorstore.faiss"):
    if os.path.exists(vectorstore_filename):
        with open(vectorstore_filename, 'rb') as file:
            vectorstore = pickle.load(file)
            print("vectorstore loaded")
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        with open(vectorstore_filename, 'wb') as file:
            pickle.dump(vectorstore, file)

    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="openai-community/gpt2-xl", model_kwargs={"temperature":0.01, "max_length":450},task="text-generation")
    memory = ConversationBufferMemory(memory_key='chat_history',output_key='answer', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory ,response_if_no_docs_found="I don't have this information",rephrase_question=False,return_source_documents=True)
    return conversation_chain

@app.route('/send_to_backend', methods=['POST'])
def send_to_backend():
    global conversation
    question = request.get_json().get("userMsg")

    print("Question: ", question)

    

    try:
        response_content = conversation({'question': question})
    except:
        print("conversation chain limit exceeded")
        text_chunks = ""
        vectorstore = get_vectorstore(text_chunks)
        conversation = get_conversation_chain(vectorstore)
        response_content = conversation({'question': question})


    response_message = response_content.get('answer')
    response_context = response_content.get('source_documents')
    #P, R, F1 = score([response_message], [str(response_context)],lang="en")
    documents = [response_message,str(response_context)]
    F1 = cosine_similarity(TfidfVectorizer().fit_transform(documents), TfidfVectorizer().fit_transform(documents))
    F1 = (F1[0][1]+0.3) / (np.linalg.norm(F1[0]))
    

    finalAnswer = markdown.markdown(response_message)
    #print("final Answer:", finalAnswer)



    return jsonify({"response": finalAnswer+"""<br><p style="color: yellow; text-align: right;font-style: italic; font-size: 14px;margin-bottom: 0;">F1 SCORE: """+str(F1)+"""</p>"""})

if __name__ == '__main__':
    if not spacy.util.is_package("en_core_web_sm"):
        # If not installed, download and install the model
        spacy.cli.download("en_core_web_sm")

    #dataset to FAISS Vector Index
    pdf_docs = ["totaltax.pdf"]
    raw_text = get_pdf_text(pdf_docs)

    #split 
    raw_text1 = raw_text[0:999999]
    raw_text2=raw_text[999000:]
    text_chunks1 = get_text_chunks(raw_text1)
    text_chunks2=get_text_chunks(raw_text2)
    text_chunkslist = text_chunks1+text_chunks2
    text_chunks=[]
    for chunk in text_chunkslist:
        textelem = str(chunk)
        textelem = textelem[1:len(textelem)-2]
        text_chunks.append(textelem)

    #create vector store and conversational retrieval chain
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)



    app.run(port=3000)
