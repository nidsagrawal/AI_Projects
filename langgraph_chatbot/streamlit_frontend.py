import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

st.markdown(""" <style> .stApp { background-color: rgb(178, 217, 118) }</style>""", unsafe_allow_html=True)
# creating empty list of dict to maintain history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] =[]

config = {'configurable': {'thread_id':'thread-1' }}

# loading all previous messages
for message in st.session_state['chat_history']:
  with st.chat_message(message['role']):
    st.text(message['content'])

# getting user input
user_input = st.chat_input('Type here')


if user_input:
   st.session_state['chat_history'].append({'role':'user', 'content': user_input})
   with st.chat_message('user'):
    st.write(user_input)

   response = chatbot.invoke({'messages':[HumanMessage(content = user_input)]},config=config)
   ai_message = response['messages'][-1].content[-1]['text']
   st.session_state['chat_history'].append({'role':'assistant', 'content': ai_message})
   with st.chat_message('assistant'):
    st.write(ai_message) 

