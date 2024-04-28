import time
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together

from footer import footer

# Configure Streamlit page settings and theme
st.set_page_config(page_title="NEETEE", layout="centered")

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, your task is to provide accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in bullet points to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question.
- Clarify the general applicability of the legal rules or sections mentioned.
- Limit responses to essential information that directly addresses the user's question.
- Avoid assuming specific contexts not provided in the query.
- Conclude with a brief summary of the legal discussion and correct any misinterpretations.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law]
- [Provide a concise explanation of how the law is typically interpreted]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information]
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

api_key = os.getenv('1e952d8a04e6de74561547a18e57ed5250fdd9e')
llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key=api_key)

qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=st.session_state.memory, retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})

def extract_answer(full_response):
    """Extract the answer from the LLM's full response."""
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        answer_end = len(full_response)
        return full_response[answer_start:answer_end].strip()
    return full_response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

input_prompt = st.chat_input("Say something...")
if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")

    st.session_state.messages.append({"role": "user", "content": input_prompt})
    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            answer = extract_answer(result["answer"])

            # Initialize the response message
            full_response = "⚠️ **_Please verify for accuracy._** \n\n\n"
            for chunk in answer:
                # Simulate typing
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button('🗑️ Reset All Chat', on_click=reset_conversation):
            st.experimental_rerun()