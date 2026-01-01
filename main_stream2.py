from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic import hub
from langchain_classic.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import os
import streamlit as st
import tempfile

# 배포시---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# load_dotenv()
api_key = st.text_input("OpenAI_API_KEY",type='password')
# 제목 
st.title("ChatPDF")
st.write("---")
# streamlit 업로드 설정
uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type=['pdf'])
st.write('---')

# 업로드한 파일 불러오기 
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory() # 임시폴더 생성
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)#임시폴더에서 업로드된 pdf로딩
    pages = loader.load_and_split()
    return pages
  
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)


  #Splitter
  text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
      chunk_size=100,
      chunk_overlap=20,
      length_function=len,
      is_separator_regex=False,
  )

  texts = text_splitter.split_documents(pages)

  #Embedding
  embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",api_key=api_key)

  #Chroma DB
  db = Chroma.from_documents(texts,embeddings_model)
  
  # 배포시---
  import chromadb
  chromadb.api.client.SharedSystemClient.clear_system_cache()
  
  llm = ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=api_key)

  # Chroma 벡터 저장소에 대한 Retriever 생성
  retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=llm
  )
  # 스트리밍 처리할 Handler 생성
  class StreamHandler(BaseCallbackHandler):
      def __init__(self, container, initial_text=""):
          self.container = container
          self.text=initial_text
      def on_llm_new_token(self, token: str, **kwargs) -> None:
          self.text+=token
          self.container.markdown(self.text)
          
           
  # 사용자 질문
  st.header('PDF에게 질문해보세요!!!')
  question = st.text_input("질문을 입력하세요")
 
  if st.button("질문하기"):
    with st.spinner("Wait for it..."):

      #Prompt Template
      prompt = hub.pull('rlm/rag-prompt')

      #docs 검색결과 format
      def formart_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
      
      # 출력공간 확보 stream 부분
      chat_box = st.empty()
      stream_handler = StreamHandler(chat_box)             
      generate_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0, openai_api_key=api_key, streaming=True, callbacks=[stream_handler])

      #rag chain
      rag_chain = (
        {"context":retriever_from_llm|formart_docs,"question":RunnablePassthrough()}
        | prompt
        | generate_llm
        | StrOutputParser()
      )
      result = rag_chain.invoke(question)
