from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchainhub import Client
from langchain_core.prompts import PromptTemplate

load_dotenv()


def main():
    pdf_path = "/home/haji/LangChain/langchain-course/space-booklet-english-2015.pdf"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retriever = new_vectorstore.as_retriever()

    prompt_template = """Use the following context to answer the question as concisely as possible.
    Context: {context}
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
    )

    llm = ChatOllama(model="llama3.2")

    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | {"answer": lambda res: res.content}
    )

    res = rag_chain.invoke("Give me 4 facts about neptune")
    print(res["answer"])


if __name__ == "__main__":
    main()
