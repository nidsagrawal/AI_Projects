from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(model='claude-sonnet-5')


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# in this function we are invoking the llm with the messages , the only difference is
#  we are filtering the response to get only ai message content and ignoring other things in response
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else response

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = " ".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        text = str(content)

    return {"messages": [text]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
