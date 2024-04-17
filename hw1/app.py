from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.messages import (
        SystemMessage, 
        HumanMessage, 
)
import traceback


def question_answer(llm, query):
    prompt = [
        SystemMessage(content="""You are a bot that helps answer questions about political figures. Find the relevant information about the political figure the user gives, 
            then answer their question. Keep your answer concise and limit your answer to three sentences. If you do not know how to answer a question, just say that you don't know."""),
        HumanMessage(content=query),
        ]
    return prompt



def main():
    llm = GoogleGenerativeAI(model="gemini-pro")


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
        



if __name__ == "__main__":
    main()
