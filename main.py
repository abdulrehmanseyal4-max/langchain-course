import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retreiving...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    llm = ChatOllama(model="llama3.2")

    query = "What is pinecone in machine learning"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    
    # Print the result
    if hasattr(result, 'content'):
        print(result.content)
    else:
        print(result)

    vectorStore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )
    retrival_chain = create_retrieval_chain(
        retriever=vectorStore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    resulta = retrival_chain.invoke(input={"input": query})

    print("\n" + "="*50)
    print("RAG Retrieved Answer:")
    print("="*50)
    
    # The retrieval chain returns a dict with 'answer' and 'context' keys
    if isinstance(resulta, dict):
        print(resulta.get('answer', resulta))
    elif hasattr(resulta, 'content'):
        print(resulta.content)
    else:
        print(resulta)

    template = """
        Use the following pieces of context to answer the question at the end.
        if you don't know the answer. just say that you dont know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of an answer.

        {context}

        Question: {question}

        Helpful Answer:
    """
    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vectorStore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    if isinstance(resulta, dict):
        print(resulta.get('answer', resulta))
    elif hasattr(resulta, 'content'):
        print(resulta.content)
    else:
        print(resulta)