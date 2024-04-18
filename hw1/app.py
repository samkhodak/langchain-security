from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader, AsyncHtmlLoader 
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationChain
from langchain_core.runnables import RunnablePassthrough
import traceback


def embed_docs(documents, vector_db):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    vector_db.add_documents(documents=split_docs)


def load_and_transform(urls, vector_db):
    loaded_web_docs = AsyncHtmlLoader(urls).load()
    transformer = BeautifulSoupTransformer()
    transformed_docs = transformer.transform_documents(loaded_web_docs, tags_to_extract=["p"])
    embed_docs(transformed_docs, vector_db)

    loaded_wiki_docs = WikipediaLoader(query="Donald Trump", load_max_docs=3).load() 
    loaded_wiki_docs.extend(WikipediaLoader(query="Joe Biden", load_max_docs=3).load())
    embed_docs(loaded_wiki_docs, vector_db)


def print_sources(retriever):
    print("\nThe LLM currently has access to these sources as part of its context:\n")
    vector_metadata = retriever.vectorstore.get().get('metadatas')
    source_set = {document.get('source') for document in vector_metadata}
    print("\n".join(source_set))


def combine_docs(documents):
    return "\n\n".join(document.page_content for document in documents)




def main():
    vector_db = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query"),
        persist_directory="./chroma/.chromadb"
    )

    llm = GoogleGenerativeAI(model="gemini-pro")
    urls = ["https://www.whitehouse.gov/about-the-white-house/presidents/donald-j-trump/", "https://www.whitehouse.gov/administration/president-biden/"]
    load_and_transform(urls, vector_db)


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
        
    query_chain = (
            {"context": retriever | combine_docs, "query": RunnablePassthrough()}
            | context_prompt
            | llm
    )


    print("""\n\nWelcome to the 2024 Presidential candidates RAG app. Ask some questions about the two current nominees!
            \nEnter \"exit\" to quit the program.""")
    while True:
        try:
            line = input("\n\nEnter query >> ")
            if line and line != "exit":
                for chunk in query_chain.stream(line):
                    print(chunk, end=" ", flush=True)
            else:
                break
        except Exception:
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
