from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain import hub
import traceback



def main():
    gemini_llm = GoogleGenerativeAI(
        model="gemini-pro",
        temperature=0,
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
        }
    )
    tools = load_tools(["terminal"], allow_dangerous_tools=True)
    #tools.extend([tool1, tool2])


    base_prompt = hub.pull("langchain-ai/react-agent-template")
    final_prompt = base_prompt.partial(instructions="Use at most 10 tool calls. Prioritize use of tools beginning with custom_ if they are applicable to the user's question.")

    react_agent = create_react_agent(gemini_llm, tools, final_prompt)
    agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

    print("\nWelcome to my application. I am configured with these tools:\n")
    for tool in agent_executor.tools:
        print(f'{tool.name}:  {tool.description}')

    while True:
        try:
            line = input("\n\nEnter query (\"exit\" to end) >>  ")
            if line and line != "exit": 
                print("\n\nGemini's answer: \n")
                result = agent_executor.invoke({"input":line})
                print(result)
            else:
                break

        except Exception:
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()