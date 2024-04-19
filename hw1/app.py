from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader, AsyncHtmlLoader 
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import traceback



def embed_docs(documents, vector_db) -> None:
    """
    Splits documents into chunks and creates embeddings with vector store.
    :param documents: list of Document objects
    :param vector_db: Chroma object
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    vector_db.add_documents(documents=split_docs)


def load_and_transform(vector_db) -> None:
    """
    Retrieves document sources from text files, loads and extracts text from
    them, then embeds and adds to vector store.
    :param vector_db: Chroma object
    """
    urls = retrieve_file("./sources/urls/urls.txt")
    wiki_topics = retrieve_file("./sources/wiki/wiki-pages.txt")
    loaded_web_docs = AsyncHtmlLoader(urls).load()
    transformer = BeautifulSoupTransformer()
    transformed_docs = transformer.transform_documents(loaded_web_docs, tags_to_extract=["p"])
    embed_docs(transformed_docs, vector_db)

    for topic in wiki_topics:
        loaded_wiki_docs = WikipediaLoader(query=topic, load_max_docs=3).load() 
        embed_docs(loaded_wiki_docs, vector_db)


def print_sources(retriever) -> None:
    """
    Displays web sources for each Document in vector store, using retriever.
    :param retriever: VectorStoreRetriever object
    """
    print("\nThe LLM currently has access to these sources as part of its context:\n")
    vector_metadata = retriever.vectorstore.get().get('metadatas')
    source_set = {document.get('source') for document in vector_metadata}
    print("\n".join(source_set))


def combine_docs(documents) -> str:
    """
    Combines documents together into newline-separated string.
    :param documents: list of Document objects
    :return: str
    """
    return "\n\n".join(document.page_content for document in documents)


def retrieve_file(file_path) -> list:
    """
    Opens file and creates an array filled with a string from each line.
    :param file_path: str
    :return: list of strings
    """
    with open(file_path, 'r') as file:
        file_contents = [line.rstrip('\n') for line in file] #Strip newline from end of each line
        return file_contents
     




def main():
    vector_db = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query"),
        persist_directory="./chroma/.chromadb"
    )

    gemini_llm = GoogleGenerativeAI(
            model="gemini-pro",
            temperature=0,
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
    )

    gpt_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

   
    load_and_transform(vector_db)


    retriever = vector_db.as_retriever()
    print_sources(retriever)


    context_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a bot that helps answer questions about political figures. Find the relevant 
        information about the political figure the user gives, then answer their question. In searching 
        for accurate information, use only the provided context. Limit your answer to ten sentences. If 
        you do not know how to answer a question, just say that you don't know.
        Provided context: {context}"""),
        ("human", "{query}"),
    ])
       
    # Set up a chain for each model, using the same context.
    gemini_query_chain = (
            {"context": retriever | combine_docs, "query": RunnablePassthrough()}
            | context_prompt
            | gemini_llm
            | StrOutputParser()
    )

    gpt_query_chain = (
            {"context": retriever | combine_docs, "query": RunnablePassthrough()}
            | context_prompt
            | gpt_llm
            | StrOutputParser()
    )

    
    
    print("""\n\nWelcome to the 2024 Presidential candidates RAG app. Ask some questions about the two current nominees!
            \nEnter \"exit\" to quit the program.""")
    while True:
        try:
            # Invoke a call for each model to answer the same query. 
            line = input("\n\nEnter query >> ")
            if line and line != "exit": 
                print("\n\nGemini's answer: \n")
                for chunk in gemini_query_chain.stream(line):
                    print(chunk, end=" ", flush=True)

                # This GPT model may not support streaming.
                print("\n\nGPT's answer: \n")
                response = gpt_query_chain.invoke(line)
                print(response)
            else:
                break

        except Exception:
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()
