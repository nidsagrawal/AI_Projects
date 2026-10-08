import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


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

   response_generator = chatbot.stream(
     {'messages':[HumanMessage(content = user_input)]},
     config=config,
     stream_mode='messages')
   
   with st.chat_message('assistant'):
         ai_message = st.write_stream(message_chunk.content for message_chunk,metadata in response_generator) 
   

   st.session_state['chat_history'].append({'role':'assistant', 'content': ai_message})
   

