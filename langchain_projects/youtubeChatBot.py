from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

llm= ChatAnthropic(model='claude-sonnet-5')
try:
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id= "Gfr50f6ZBvo")
    transript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions available")


splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap = 200)
chunks = splitter.create_documents([transript])    
print(len(chunks))
# embeddings =vo.embed(texts =chunks,model = "voyage-4-large" ,input_type="document")
embeddings = HuggingFaceEmbeddings()
vector_store =Chroma.from_documents(chunks,embeddings,collection_name="chatbot_dir",persist_directory='my_chroma_db1')

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs ={"k":2})


prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
result =main_chain.invoke('Can you summarize the video')

print(result)

