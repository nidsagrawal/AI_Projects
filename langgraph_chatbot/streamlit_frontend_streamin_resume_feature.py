import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage,AIMessage
import uuid

# **********************utility functions ************************

# generate new thread_id function utility
def generate_thread_id():
   thread_id = uuid.uuid4()
   return thread_id

# function to reset the chat for new chat
def reset_chat():
   thread_id = generate_thread_id()
   st.session_state['thread_id'] = thread_id
   add_thread_id(thread_id)
   st.session_state['chat_history'] =[]


def add_thread_id(thread_id):
   if thread_id not in st.session_state['chat_threads']:
      st.session_state['chat_threads'].append(thread_id)

def get_message_history(thread_id):
    state=chatbot.get_state(config={'configurable': {'thread_id':thread_id }})
    return state.values.get('messages', [])
# **********************session state setup implimentation ************************

# creating empty list of dict to maintain history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] =[]

if 'thread_id' not in st.session_state:
   st.session_state['thread_id']= generate_thread_id()

if 'chat_threads' not in st.session_state:
   st.session_state['chat_threads'] =[]
   
add_thread_id(st.session_state['thread_id'])

# **********************sideBar implementation************************

st.sidebar.title('Chatbot')

if st.sidebar.button('New Chat'):
   reset_chat()

st.sidebar.header('My conversations')


for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = get_message_history(thread_id)
        temp_messages =[]
        for msg in messages:
           if isinstance(msg,HumanMessage):
              role ='user'
           else:
              role = 'assistant'
           temp_messages.append({'role': role,'content':msg.content}) 
        st.session_state['chat_history'] =temp_messages


# ********************************************************************


# loading all previous messages
for message in st.session_state['chat_history']:
  with st.chat_message(message['role']):
    st.text(message['content'])

# getting user input
user_input = st.chat_input('Type here')

# ***************************main logic*************************************
if user_input:
  
   st.session_state['chat_history'].append({'role':'user', 'content': user_input})
   with st.chat_message('user'):
    st.write(user_input)

  # attach session threading
   config = {'configurable': {'thread_id':st.session_state['thread_id'] }}

# implement streaming
   response_generator = chatbot.stream(
     {'messages':[HumanMessage(content = user_input)]},
     config=config,
     stream_mode='messages')

   # first add the message to message_history
   with st.chat_message("assistant"):
       def ai_only_stream():
           for message_chunk, metadata in response_generator:
               if isinstance(message_chunk, AIMessage):
                   # yield only assistant tokens
                   yield message_chunk.content

       ai_message = st.write_stream(ai_only_stream())

   st.session_state['chat_history'].append({'role':'assistant', 'content': ai_message})
   

