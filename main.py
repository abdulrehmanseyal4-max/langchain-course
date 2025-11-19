from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

def main():
    print("Hello from langchain-course!")
    pdf_path = "/home/haji/LangChain/langchain-course/space-booklet-english-2015.pdf"


if __name__ == "__main__":
    main()
