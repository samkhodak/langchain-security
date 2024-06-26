from langchain_google_genai import GoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langsmith import Client
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from textwrap import dedent
from subprocess import run
import requests
import socket
import shlex
import validators
import dns.resolver as resolver
import traceback
import os




os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"gensec-hw6"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
client = Client()

gpt_llm = ChatOpenAI(model='gpt-4o', temperature=0)




class IPv4Input(BaseModel):
    """ 
    For checking correctness of IPv4 address passed to tools.

    :param address: IPv4 address, without CIDR notation.
    :type address: str
    """
    address: str = Field(description="Should be an IP address such as 208.91.197.27 or 143.95.239.83, with no CIDR notation.")
    @validator('address')
    def is_ip_address(cls, value: str) -> str:
        if not validators.ip_address.ipv4(value):
            raise ValueError("Malformed IP address")
        return value


class HostNameInput(BaseModel):
    """For checking correctness of DNS hostname passed to tools. 

    :param hostname: DNS hostname
    :type hostname: str
    """
    hostname: str = Field(description="Must be a hostname such as www.google.com")
    @validator('hostname')
    def is_dns_address(cls, value) -> str:
        if not validators.domain(value):
            raise ValueError("Malformed hostname")
        return value


@tool("retrieve_DNS_host", args_schema=IPv4Input, return_direct=False)
def retrieve_DNS_host(ip_address):
    """
    Given an IPv4 address, returns DNS hostname associated with it.
    """
    try:
        hostname, _, _ = socket.gethostbyaddr(ip_address)
        return hostname
    except socket.herror:
        raise ValueError("The IP address is not valid. Please enter an IPv4 address with no CIDR notation.")


@tool("ip_location_info", args_schema=IPv4Input, return_direct=False)
def ip_location_info(ip_address):
    """
    Get relevant location and organization information for an IP address.
    """
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_info = {
        'location': f"{response.get('city','N/A')}, {response.get('region','N/A')} - {response.get('country_name','N/A')} ({response.get('continent_code','N/A')})",
        'coordinates': f"Latitude: {response.get('latitude', 'N/A')} - Longitude: {response.get('longitude', 'N/A')}",
        'organization': response.get('org', 'N/A')
    }
    return location_info


@tool("retrieve_ip", args_schema=HostNameInput, return_direct=False)
def retrieve_ip(hostname):
    """
    Given a DNS hostname, retrieve the IP associated with it.
    """
    try:
        response = socket.gethostbyname(hostname)
        return response
    except socket.gaierror:
        raise ValueError("The hostname is not valid. Please enter a valid URL or DNS hostname.")


@tool("retrieve_DNS_records", args_schema=HostNameInput, return_direct=False)
def retrieve_DNS_records(hostname):
    """
    Needs a DNS hostname. Will retrieve relevant DNS records with a dig (A, AAAA, NS, MX).
    """
    record_types = ['A', 'AAAA', 'NS', 'MX']
    final_records = ""
    for record_type in record_types:
        try:
            resolved_info = resolver.resolve(hostname, record_type).response
            records = resolved_info.resolve_chaining().answer
            final_records += f"\n{str(records)}\n"
        except resolver.NoAnswer:
            final_records += f"No {record_type} record for this hostname.\n\n"

    return final_records


@tool("ping_host", args_schema=IPv4Input, return_direct=False)
def ping_host(ip_address):
    """
    Given an IPv4 address, will ping it and return the ping output. 
    """
    command = "ping"
    flag = "/n" if os.name == 'nt' else "-c"
    address = shlex.split(ip_address)[0]
    completed_process = run([command, flag, "2", address], capture_output=True, timeout=2)
    output = completed_process.stdout.decode('utf-8')

    return output




def main():
    base_prompt = PromptTemplate.from_template(dedent("""
        {instructions}

        You have access to the following tools:

        {tools}

        To use a tool, please use the following format:

            Thought: Do I need to use a tool? Yes
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

            Thought: Do I need to use a tool? No
            Final Answer: [your response here]
        
        Do not attach backticks to any of your outputs, including thoughts, tool calls and final answers.

        Begin!

        New input: {input}

        {agent_scratchpad} 
    """))

    prompt = base_prompt.partial(instructions=dedent("""You are an agent that is used for helping the user get IP and DNS information.
        Be as helpful as possible. If you are unable to produce an answer that is helpful to the user, say so.
        The user is allowed to look up information with SERPAPI related to IP and DNS queries ONLY. Deny them in any other case.
        Because your tools provide a lot of dense information, structure your final friendly response by separating each tool 
        call's answer in a visually pleasing list, with proper whitespace.
        If one of your tools requires a DNS name or IP address but the user provides the wrong type, use the retrieve_ip and retrieve_dns_host tools to get the right input."""))

    tools = load_tools(["serpapi"])
    tools.extend([retrieve_DNS_host, ip_location_info, retrieve_ip, retrieve_DNS_records, ping_host])

    gpt_agent = create_react_agent(gpt_llm, tools, prompt)
    gpt_executor = AgentExecutor(
            agent=gpt_agent, 
            tools=tools, 
            max_iterations=5, 
            verbose=True
    )



    print("\n\nThis agent is equipped with multiple tools that help you find information on IP addresses and DNS names.\n\n")
    for tool in gpt_executor.tools:
        print(f"\n{tool.name}: \n\n\t{tool.description}")
    

    while True:
        try:
            line = input("\n\nEnter query (\"exit\" to end) >>  ")
            if line and line != "exit": 
                print("\n\n\nPlease wait while the Agent completes your request.\n\n\n")
                result = gpt_executor.invoke({"input":line})
                print(f"\n\n{result.get('output')}")
            else:
                break

        except ValueError as v_error:
            print(f"\n\n{str(v_error)}")
        except Exception:
            traceback.print_exc()

    return


if __name__ == "__main__":
    main()
