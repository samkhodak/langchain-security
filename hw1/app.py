from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader, AsyncHtmlLoader 
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import traceback


def question_prompt(llm, query):
    prompt = [
        SystemMessage(content="""You are a bot that helps answer questions about political figures. Find the relevant 
        information about the political figure the user gives, then answer their question. Limit your answer to ten 
        sentences. If you do not know how to answer a question, just say that you don't know."""
        ),
        HumanMessage(content=query),
    ]
    return prompt

def embed_docs(documents, vector_db):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    vector_db.add_documents(documents=split_docs)



def load_and_transform(urls, vector_db):
    loaded_web_docs = AsyncHtmlLoader(urls).load()
    transformer = BeautifulSoupTransformer()
    transformed_docs = transformer.transform_documents(loaded_web_docs, tags_to_extract=["p"])
    #for page in transformed_docs:
    #    print(page.page_content)
    embed_docs(transformed_docs, vector_db)

    loaded_wiki_docs = WikipediaLoader(query="Donald Trump", load_max_docs=3).load() 
    loaded_wiki_docs.extend(WikipediaLoader(query="Joe Biden", load_max_docs=3).load())
    embed_docs(loaded_wiki_docs, vector_db)

    retriever = vector_db.as_retriever()

    #docs = retriever.get_relevant_documents("Donald Trump")
    #print(docs)
    print("SAVED DOCUMENTS -> \n")
    document_data_sources = set()
    for doc_metadata in retriever.vectorstore.get()['metadatas']:
        document_data_sources.add(doc_metadata['source'])
    for doc in document_data_sources:
        print(f"  {doc}")




def main():
    vector_db = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query"),
        persist_directory="./chroma/.chromadb"
    )
    llm = GoogleGenerativeAI(model="gemini-pro")
    urls = ["https://www.whitehouse.gov/about-the-white-house/presidents/donald-j-trump/", "https://www.whitehouse.gov/administration/president-biden/"]
    load_and_transform(urls, vector_db)

    while True:
        try:
            line = input("\n\nEnter query >> ")
            if line:
                for chunk in llm.stream(question_prompt(llm,line)):
                    print(chunk, end=" ", flush=True)
            else:
                break
        except Exception:
            traceback.print_exc()
            break



if __name__ == "__main__":
    main()
