# Import GraphSaver to save the workflow graph as a PNG
from write_graph.write_graph_to_a_file import GraphSaver
from dotenv import load_dotenv
import os

load_dotenv('../.env')  # Load environment variables from .env file

# Set USER_AGENT environment variable if not already set
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "MyLangGraphBot/1.0"


from langchain.tools import Tool

from support_portal_cases_search import get_case_descriptions_wrapper


from langchain_community.tools.tavily_search import TavilySearchResults

##############################################
# BT - Langchain community tools.
##############################################
search_internet_tool = TavilySearchResults(max_results=2)

from vectorstore_builder_class import VectorstoreBuilder

vs = VectorstoreBuilder(pdf_directory="./docs", persist_directory="./chroma_db")

retriever_tool = vs.get_retriever_tool()

tools = [retriever_tool, search_internet_tool]

# tools = [get_case_descriptions_wrapper]

# Use the retriever_tool in your agent/toolchain


from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

from langchain_core.prompts import PromptTemplate



from langchain_core.messages import SystemMessage


SYSTEM_MSG = SystemMessage(
    content="""
            You are an expert Multitech Technical Support assistant for question-answering tasks.

            1. Use only the provided Multitech internal documents to answer the question.
            2. If internal context is insufficient, use the Tavily search tool. When using the Tavily search tool, make sure it is
               related to Multitech products. For example, the product names are MTCDT, MTCDTIP, MTCAP, MTCAP2, MTCAP3, xDot, mDot...etc.
            3. Do not mix information from different sources — clearly state if the answer is from internal documents or web search.
            4. Look for the keyword from the user questions and try to match it with an exact word from the search Multitech internal documents or
               a similar word. For example, if the keyword is 'at+pp', the exact is 'at+pp' or the similar word is 'at+ppxxx', the xxx can be any
               any characters from [a to z or A to Z, 1 - 9]...etc.
            5. Provide complete, step-by-step instructions. Include links, screenshots, or examples exactly as shown.
            6. Cite the source (e.g., document name or URL).
            7. If no answer is found, say: "I don’t know."
            """
)


### Edges

# Import the GradeDocumentEdges class
from grade_document_edges import GradeDocumentEdges

# Instantiate the class
grade_document_edges = GradeDocumentEdges()

# Use the class method as the callback
def grade_documents(state):
    return grade_document_edges.grade_documents(state)



# Import the Agent class and use it for the agent node
from agent import Agent

# Instantiate the Agent class
agent_instance = Agent(SYSTEM_MSG, tools)

# Use the class method as the callback
def agent(state):
    return agent_instance.agent(state)



# Import the RewriteAgent class and use it for the rewrite node
from rewrite_agent import RewriteAgent

# Instantiate the RewriteAgent class
rewrite_agent_instance = RewriteAgent()

# Use the class method as the callback
def rewrite(state):
    return rewrite_agent_instance.rewrite(state)

# Import the GenerateAgent class and use it for the generate node
from generate_agent import GenerateAgent

# Instantiate the GenerateAgent class
generate_agent_instance = GenerateAgent()

# Use the class method as the callback
def generate(state):
    return generate_agent_instance.generate(state)


from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
# workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("use_tools", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    # BT - If the LLM responses a 'tool_calls', then tool_condition will return 'tools'
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        ##############################################################################################
        # BT - The return 'tools' from tools_condition it will invoke the "retrieve" tool in 
        #      call the RAG.
        ##############################################################################################
        # "tools": "retrieve",
        "tools": "use_tools",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    # "retrieve",
    "use_tools",
    ########################################################################################################
    # BT - When 'retrieve' return, it will run the 'grade_documents'. The 'grade_documents' will rate
    #      the message and it will return either 'generate' or 'rewrite'. 
    #      If it is 'rewrite' then the 'rewrite' will back to 'agent'. Otherwise, if it is 'generate',
    #      then it will goes to END
    ########################################################################################################
    # Assess agent decision
    grade_documents,
)
######################################################################################################
# BT - So,down here is where the decision is made based on the grading of the documents.
#      if the grade_documents returns 'generate', then it will go to END.
#      otherwise, it will go to 'rewrite' and then back to 'agent'.
######################################################################################################
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
# graph = workflow.compile()

##########################################################################
# BT - Memory Saver SECTION: Save the state of the graph for streaming.
##########################################################################
from langgraph.checkpoint.memory import MemorySaver

# Initialize MemorySaver for in-memory state management
memory = MemorySaver()


# Compile the workflow with the memory checkpointer
graph = workflow.compile(checkpointer=memory)

# Save the workflow graph as a PNG
# graph_saver = GraphSaver(graph)
# result = graph_saver.save_graph()
# print("Graph save result:", result)

# Configuration for the workflow execution
config = {"configurable": {"thread_id": "1"}}


#####################################
#BT - STREAMLIT SECTION:
#####################################
from langchain_core.messages import HumanMessage, AIMessage

def convert_messages(messages):
    converted = []
    for msg in messages:
        if msg["role"] == "user":
            converted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            converted.append(AIMessage(content=msg["content"]))
    return converted

import streamlit as st


st.title("BT - Multitech Chatbot")

# Sidebar file uploader for user documents
st.sidebar.title("Upload Your Document")
docs_folder = "./docs"
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

uploaded_file = st.sidebar.file_uploader("Choose a file to upload (PDF, TXT, etc.)", type=["pdf", "txt", "docx","md", "csv", "xlsx"])
if uploaded_file is not None:
    save_path = os.path.join(docs_folder, uploaded_file.name)
    if os.path.exists(save_path):
        st.sidebar.warning(f"A file named '{uploaded_file.name}' already exists in the docs folder. Upload aborted.")
    else:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded: {uploaded_file.name}")

# List and delete docs in the folder
st.sidebar.markdown("---")
st.sidebar.subheader("Docs in Folder:")
# Use a wider sidebar and set button width via CSS
# Use a wider sidebar and set button width via CSS
st.markdown(
    """
    <style>
    [data-testid=\"stSidebar\"] {
        min-width: 400px !important;
        max-width: 650px !important;
        padding-right: 0px !important; /* Increase padding to push scrollbar further right */
    }
    .stButton > button {
        width: 70px !important;
        white-space: nowrap;
        margin-right: 0px !important;
        margin-left: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
doc_files = [f for f in os.listdir(docs_folder) if os.path.isfile(os.path.join(docs_folder, f))]
for doc in doc_files:
    col1, col2 = st.sidebar.columns([4, 1], gap="small")  # Move button to the right
    with col1:
        st.write(doc)
    with col2:
        if st.button("Delete", key=f"delete_{doc}"):
            os.remove(os.path.join(docs_folder, doc))
            # Remove from vectorstore as well
            from vectorstore_builder_class import VectorstoreBuilder
            vs = VectorstoreBuilder(pdf_directory=docs_folder, persist_directory="./chroma_db")
            vs.delete_file_from_vectorstore(doc)
            # Remove doc from processed_files.json
            processed_files_path = vs.processed_files_record
            if os.path.exists(processed_files_path):
                import json
                with open(processed_files_path, "r") as f:
                    processed_files = set(json.load(f))
                if doc in processed_files:
                    processed_files.remove(doc)
                    with open(processed_files_path, "w") as f:
                        json.dump(list(processed_files), f)
            st.rerun()

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        for event in graph.stream({"messages": [{"role": "user", "content": prompt}] + st.session_state.messages},config):
        # converted_history = convert_messages(st.session_state.messages)
        # for event in graph.stream({"messages": converted_history}, config):
            # print('BT - What is this: ', event)
            if 'retrieve' not in event:
                for value in event.values():
                    st.markdown(value["messages"][-1].content)
                    print("Assistant:", value["messages"][-1].content)
                    response += value["messages"][-1].content
        
        st.session_state.messages.append({"role": "assistant", "content": response})
