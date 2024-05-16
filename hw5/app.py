from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from langsmith import Client
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import traceback
import time
import asyncio
import os


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"gensec-hw5"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()


gemini_llm = GoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0,
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    }
)
gpt_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# Maximum tokens allowed for a prompt, otherwise the chain call will be rejected.
MAX_PROMPT_TOKENS = 500



def load_code(file_name) -> str:
    """
    Loads a file from current directory, parsing the file into code and returning it as a string.
    :param file_name: name of a file, including extension.
    :type file_name: str
    :return: string of code from loaded documents
    :rtype: str
    """
    loader = GenericLoader.from_filesystem(
            path=f"sources/{file_name}",
            glob="*",
            suffixes=[".c", ".cpp"],
            parser=LanguageParser(parser_threshold=1000),
    )

    docs = loader.load()
    if not docs:
        raise ValueError("The filename was not found. Try again.")

    document_code = "\n\n\n".join([document.page_content for document in docs])
    return document_code


async def run_code_chain(document, chain):
    response = await chain.ainvoke(document)
    return response


async def dual_code_chains(document, gemini_chain, gpt_chain):
    async with asyncio.TaskGroup() as tg:
        gemini_result = tg.create_task(run_code_chain(document, gemini_chain))
        gpt_result = tg.create_task(run_code_chain(document, gpt_chain))

    print(gemini_result.result())
    print(gpt_result.result())






def main():
    code_chain = lambda prompt, llm: (
    {"code_content":RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser()
    )

    prompt = PromptTemplate.from_template("""You are an expert at spotting security vulnerabilities and bad practice in C and C++ code.
        Your job is to take in a segment of code and identify the top three vulnerabilities that can be found in it. 
        Keep your answer concise, but list out the top three results with enough detail that the reader can understand how to solve the security issue.
        Your answer should be in the following format:
        Here are three of the top security vulnerabilities in the provided [language] code.
        1. [vulnerability 1]
        ... and so on.
        Code content: {code_content}
    """)

    gemini_chain = code_chain(prompt, gemini_llm)
    gpt_chain = code_chain(prompt, gpt_llm)



    while True:
        try:
            line = input("\n\nEnter query (\"exit\" to end) >>  ")
            if line and line != "exit": 
                document = load_code(line)
                if (gemini_llm.get_num_tokens(document) > MAX_PROMPT_TOKENS):
                    raise RuntimeError(f"The document you are trying to check is too large for the {MAX_PROMPT_TOKENS} token limit.")
                
                start = time.perf_counter()
                asyncio.run(dual_code_chains(document, gemini_chain, gpt_chain))
                time_taken = time.perf_counter() - start
                print(f"""\n\nTime taken to complete both requests: {"{:.2f}".format(time_taken)} seconds.""")

            else:
                break

        except ValueError as v_error:
            print(f"\n\n{str(v_error)}")
        except Exception:
            traceback.print_exc()


if __name__=="__main__":
    main()
