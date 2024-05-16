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
gpt_llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Maximum tokens allowed for a prompt, otherwise the chain call will be rejected.
MAX_PROMPT_TOKENS = 500



def load_code(file_name) -> str:
    """ Loads a file from current directory, parsing the file into code and returning it as a string.

    Args:
        file_name (str): name of a file, including extension

    Raises:
        ValueError: error when filename is not successfully found in filesystem.

    Returns:
        str: string of code from loaded documents
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


async def run_chain(chain, document):
    """ Asynchronously invokes one chain and returns the response. 

    Args:
        chain (LangChain RunnableSequence): LangChain chain
        document (str): String of document text to feed LLM chain.

    Returns:
        str: Result from chain
    """    
    response = await chain.ainvoke(document)
    return response


async def dual_chains(chain_one, chain_two, document):
    """ Asynchronously creates two chain tasks and executes both, gathering a tuple of results.

    Args:
        chain_one (LangChain RunnableSequence): LangChain chain
        chain_two (LangChain RunnableSequence): LangChain chain
        document (str): String of document text to feed LLM chain.

    Returns:
        Tuple[str,str]: results from both chains
    """
    async with asyncio.TaskGroup() as tg:
        result_one = tg.create_task(run_chain(chain_one, document))
        result_two = tg.create_task(run_chain(chain_two, document))

    return result_one.result(), result_two.result()






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
        You are allowed to reference functions and one-line bits of code to help the reader, but do not include code blocks in your answer. Print all code lines on a new line.
        Your answer should be in the following format:
        Here are three of the top security vulnerabilities in the provided [language] code.
        1. [vulnerability 1]
           - Issue: [explanation of vulnerability]
           - Recommendation: [explanation of solution]

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
                asyncio.run(dual_chains(gemini_chain, gpt_chain, document))
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
