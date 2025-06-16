import faiss
from uuid import uuid4
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class RAG:
    
    def __init__(self, data_path="./Data/", glob_pattern="*.txt"):
        load_dotenv()
        self.data_path = data_path
        self.glob_pattern = glob_pattern

    def load_documents(self):
        loader = DirectoryLoader(path=self.data_path, glob=self.glob_pattern, loader_cls=TextLoader)
        return loader.load()

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=0,
            separators=""
        )
        return splitter.split_documents(documents)

    def get_embeddings(self):
        return HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    def build_vector_store(self, chunks, embeddings):
        dim = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatIP(dim)
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(documents=chunks, ids=uuids)
        return vector_store

    def get_retriever(self):
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        embeddings = self.get_embeddings()
        vector_store = self.build_vector_store(chunks, embeddings)
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
