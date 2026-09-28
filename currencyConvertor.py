from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool , InjectedToolArg
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import Annotated
import requests
import json

load_dotenv()

llm= ChatAnthropic(model='claude-sonnet-5')
@tool
def get_conversion_factor(base_currency: str , target_currency: str)-> float:
    """
    This tool gets the conversion factor between given base and target currency
    """
    
    url = f"https://v6.exchangerate-api.com/v6/c0b427954f817e0e5ece6ced/pair/{base_currency}/{target_currency}"

    response= requests.get(url)

    return response.json()

@tool
def convert(conversion_factor:Annotated[float,InjectedToolArg] , base_currency_value ) -> float:
    """
    This function calculates the target currency value given that base currency value and conversion rate
    """
    return conversion_factor * base_currency_value

llm_with_tools = llm.bind_tools([get_conversion_factor,convert])

messages =[HumanMessage('What is the conversion factor between USD and INR  and based on that can you convert 10 usd to inr')]

ai_message= llm_with_tools.invoke(messages)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    if(tool_call['name'] =='get_conversion_factor'):
        tool_call1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_call1.content)['conversion_rate']
        messages.append(tool_call1)
       
ai_message1 = llm_with_tools.invoke(messages)
messages.append(ai_message1)

for tool_call in ai_message1.tool_calls:
    if (tool_call['name'] == 'convert'):
        tool_call['args']['conversion_factor'] =conversion_rate
        tool_call2 = convert.invoke(tool_call) 
        messages.append(tool_call2)  

final_response = llm_with_tools.invoke(messages)

print(final_response.content)

     
