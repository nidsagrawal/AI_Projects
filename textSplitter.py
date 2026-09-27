from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter,Language 
from langchain_community.document_loaders import PyPDFLoader

# splitter  = CharacterTextSplitter(
#         chunk_size=100,
#         chunk_overlap=0,
#         separator=''
# )

# loader = PyPDFLoader("Sangeet_List_and_Relations.pdf")

# documents = loader.load()

# text = """"The Statue of Unity is one of the most famous landmarks in India. It is located near Kevadia in Gujarat, on the banks of the Narmada River. The statue is dedicated to Sardar Vallabhbhai Patel, one of the key leaders in India's freedom struggle and the first Deputy Prime Minister and Home Minister of independent India. Standing at a height of 182 metres, it is the world's tallest statue. The statue was inaugurated on 31 October 2018, on the birth anniversary of Sardar Patel. It represents unity, strength, and the important role played by Sardar Patel in bringing together the princely states of India. Today, the Statue of Unity is a major tourist attraction and attracts visitors from India and around the world.

# """
# result = splitter.split_text(documents[0].page_content)

# print(result[0])



# -----------------------------------------------------

splitter2  = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=100, chunk_overlap=0)

text2 = """
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace 
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from typing import Literal


load_dotenv()

myendpoint = HuggingFacePipeline.from_model_id(
    model_id ="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    task="text-generation"
)   

parser = StrOutputParser()
parser2 = JsonOutputParser()
template1 =PromptTemplate(
    
    input_variables=['input'],
    template='Write detailed report on : {input}',
)

template2 =PromptTemplate(
    
    input_variables=['input'],
    template='Write 5 line summery on : {input}',
)"""
result2 = splitter2.split_text(text2)
print(result2[0])


