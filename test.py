from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace 
from langchain_openai import OpenAI
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
)

model = ChatHuggingFace(llm = myendpoint)
# ----------------------------------------------------------------------------------
# chain = template1 | model | parser | template2 | model | parser

# response = chain.invoke({'input' :'Climate change'})
# print (response.content)
# ----------------------------------------------------------------------------------
# prompt1 = template1.invoke({'input' :'Climate change'})
# response = model.invoke(prompt1)
# prompt2 = template2.invoke({'input' :response.content})
# finalResponse = model.invoke(prompt2)


# print (finalResponse.content)
# ----------------------------------------------------------------------------------
# 
# template_for_parser2 = PromptTemplate(
#     input_variables=[],
#     template='give me age, name, city of a fictional person \n {format_instructions}',
#     partial_variables={'format_instructions': parser2.get_format_instructions()}
# ) 

# chain = template_for_parser2 | model | parser2

# response = chain.invoke({})


# print (response)
# ------------------------------------------------------------------------------------

# class Person(BaseModel):
#     name: str = Field(description='Name of the person')
#     age: int = Field(gt=18, description='Age of the person')
#     city: str = Field(description='Name of the city the person belongs to')

# parser3 = PydanticOutputParser(pydantic_object=Person)

# template1_for_parser3 = PromptTemplate(
#     input_variables=[], 
#     template='generate age, name, city of a fictional person \n {format_instructions}',
#     partial_variables={'format_instructions': parser3.get_format_instructions()}        
# )

# chain = template1_for_parser3 | model 


# response = chain.invoke({})

# chain.get_graph().print_ascii()

# ----------------------------------------------------------------------------------

# temp1 = PromptTemplate(
#     input_variables=['text'],
#     template='Write short notes on : {text} '
# )

# temp2 = PromptTemplate(
#     input_variables=['text'],
#     template='Write 5 questions and answers on : {text} '
# )

# temp3 = PromptTemplate(
#     input_variables=['notes','quiz'],       
#     template='merge provided notes and quiz into single document \n notes ->{notes} \n quiz -> {quiz} '
# )

# parallelChain = RunnableParallel({
#     'notes':temp1 | model | parser,
#     'quiz':temp2 | model | parser
# }   
# )

# text = """Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

# The advantages of support vector machines are:

# Effective in high dimensional spaces.

# Still effective in cases where number of dimensions is greater than the number of samples.

# Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

# Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

# The disadvantages of support vector machines include:

# If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

# SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

# The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
# """

# mergeChain = temp3 | model | parser

# finalChain = parallelChain | mergeChain

# response = finalChain.invoke({'text':text})
 
# print(response)
# ----------------------------------------------------------------------------------

class Feedback(BaseModel):
    sentiment: Literal['positive' , 'negative'] = Field(description='Sentiment of the feedback')

parser4 = PydanticOutputParser(pydantic_object=Feedback)
template_for_parser4 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': parser4.get_format_instructions()
    }
)  

classificationChain = template_for_parser4 | model | parser4

repsonse =classificationChain.invoke({'feedback':'I am very happy with the service provided by your company. '})

print(repsonse.content)

# template3 = PromptTemplate(
   
#     template='give suitable response for given positive feedback :\n {feedback}',
#      input_variables=['feedback']
# )

# template4 = PromptTemplate(
      
#     template='give suitable response for given negative feedback : \n{feedback}',
#     input_variables=['feedback']
# )


# # branchOutput = RunnableBranch(
# #     branches={
# #         'positive': template1 | model | parser,
# #         'negative': template2 | model | parser
# #     },
# #     input_key='sentiment'
# # )

# branchOutput = RunnableBranch(
#     (lambda x:x.sentiment == 'positive', template3 | model | parser),
#     (lambda x:x.sentiment == 'negative', template4 | model | parser),
#     RunnableLambda(lambda x: "could not find sentiment")
# )
# finalChain = classificationChain | branchOutput

# response = finalChain.invoke({'feedback':'I am very happy with the service provided by your company. '})

# print(response)