import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


class RetrieverApp:
    def __init__(
        self, model_name="gpt-4o-mini", embedding_model="text-embedding-ada-002"
    ):
        load_dotenv()
        self.query = None
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.vectorstore = self._initialize_vectorstore()
        self.retrieval_chain = self._create_retrieval_chain()

    def _initialize_llm(self):
        return ChatOpenAI(model=self.model_name, temperature=0)

    def _initialize_embeddings(self):
        return OpenAIEmbeddings(model=self.embedding_model)

    def _initialize_vectorstore(self):
        return PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"], embedding=self.embeddings
        )

    def _create_retrieval_chain(self):
        prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(
            retriever=self.vectorstore.as_retriever(),
            combine_docs_chain=combine_docs_chain,
        )

    def run_query(self, query: str):
        self.query = query
        result = self.retrieval_chain.invoke(input={"input": query})
        return result


if __name__ == "__main__":
    print("Retrieving...")

    app = RetrieverApp()
    response = app.run_query("what is Pinecone in machine learning?")
    print(response["answer"])
