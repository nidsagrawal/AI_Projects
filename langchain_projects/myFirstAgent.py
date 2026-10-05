from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
import requests
import os
from langsmith import Client


load_dotenv()

search_tool = DuckDuckGoSearchRun()
client = Client()

prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True
)
# os.environ["dangerously_pull_public_prompt"] = "true"
@tool
def get_weather_data(city:str)-> str:
    """
    This function returns the weather data for given city
    """
    url = f"https://api.weatherstack.com/current?access_key=e8e6f85297f909d2bceb6abf91061c07&query={city}"
    response = requests.get(url)
    return response.json()

llm = ChatAnthropic(model='claude-sonnet-5')

agent = create_agent(
    model=llm,
    tools =[search_tool,get_weather_data],
    system_prompt=prompt.template
)

response = agent.invoke({"messages": [{"role": "user", "content": "Find the capital of Madhya Pradesh, then find it's current weather condition"}]})
print(response["messages"][-1].content)



