from sentence_transformers import SentenceTransformer
import pinecone
import openai
from openai import OpenAI
import streamlit as st
import os

OPENAI_API_KEY = "your openai api key here"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

PINECONE_API_KEY = "your pinecone api key here"
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
model = SentenceTransformer('all-MiniLM-L6-v2')

pc = pinecone.Pinecone(
    pinecone_api_key=os.environ['PINECONE_API_KEY'],
    environment='gcp-starter'
)

index = pc.Index('langchain-chatbot', host="pinecone index host here")


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=5, include_values=True, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']


def query_refiner(conversation, query):
    prompt = f"""Given the following user query and conversation log, formulate a question that would be the most 
        relevant to provide the user with an answer from a knowledge base.
        \n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string