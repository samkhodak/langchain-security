from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader, AsyncHtmlLoader 
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.messages import SystemMessage, HumanMessage
import traceback


def question_answer(llm, query):
    prompt = [
        SystemMessage(content="""You are a bot that helps answer questions about political figures. Find the relevant information about the political figure the user gives, 
            then answer their question. Keep your answer concise and limit your answer to three sentences. If you do not know how to answer a question, just say that you don't know."""),
        HumanMessage(content=query),
        ]
    return prompt

def load_and_transform(urls):
    loaded_web_docs = AsyncHtmlLoader(urls).load()
    transformer = BeautifulSoupTransformer()
    transformed_docs = transformer.transform_documents(loaded_web_docs, tags_to_extract=["p"])
    for page in transformed_docs:
        print(page.page_content)

    loaded_wikis = WikipediaLoader(query="Donald Trump", load_max_docs=3).load() 
    #print(loaded_wikis)




def main():
    llm = GoogleGenerativeAI(model="gemini-pro")
    urls = ["https://www.whitehouse.gov/about-the-white-house/presidents/donald-j-trump/", "https://www.whitehouse.gov/administration/president-biden/"]
    load_and_transform(urls)

    ("""
    while True:
        try:
            line = input("Enter query >> ")
            if line:
                result = llm.invoke(question_answer(llm,line))
                print(result)
            else:
                break
        except Exception:
            traceback.print_exc()
            break
    """)
        



if __name__ == "__main__":
    main()
