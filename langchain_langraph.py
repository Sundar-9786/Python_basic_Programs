"""

🧩 1. What is LangChain?

LangChain is a framework for developing LLM-powered applications that integrate reasoning, 
memory, tools, and data sources.

It provides modular abstractions for:

Chains (sequences of components)

Agents (autonomous reasoning units that decide which tool to use)

Memory (state persistence)

Toolkits (integration with APIs, DBs, vector stores, etc.)

Think of LangChain as a software orchestration layer that connects models and logic,
where you design your agent’s reasoning and flow manually using chains or agents.






⚙️ 2. Example — Multi-Agent Sequential Workflow in LangChain

Let’s design a workflow with 3 agents:

Research Agent — fetches information from the web.

Analysis Agent — summarizes and identifies key insights.

Report Agent — generates a final formatted report.

"""
#=================================================================
#=======================Lang Chain ===============================
#=================================================================


# pip install langchain langchain-openai langchain-community

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate

# === Base LLM ===
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# === Define tools ===
def web_search_tool(query):
    # (Stub — in production, connect to SerpAPI or Tavily)
    return f"[Simulated search result for: {query}]"

tools = [Tool(name="Web Search", func=web_search_tool, description="Useful for searching the web.")]

# === Agent 1: Research Agent ===
research_prompt = PromptTemplate(
    template="Search the web for {topic} and summarize 5 key facts.",
    input_variables=["topic"]
)
research_chain = LLMChain(llm=llm, prompt=research_prompt, output_key="research_summary")

# === Agent 2: Analysis Agent ===
analysis_prompt = PromptTemplate(
    template="Analyze the following research and extract key insights:\n{research_summary}",
    input_variables=["research_summary"]
)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analysis")

# === Agent 3: Report Agent ===
report_prompt = PromptTemplate(
    template="Generate a concise technical report based on the following analysis:\n{analysis}",
    input_variables=["analysis"]
)
report_chain = LLMChain(llm=llm, prompt=report_prompt, output_key="report")

# === Sequence Multi-Agent Workflow ===
workflow = SequentialChain(
    chains=[research_chain, analysis_chain, report_chain],
    input_variables=["topic"],
    output_variables=["report"],
    verbose=True
)

# === Run ===
result = workflow({"topic": "impact of quantum computing on cryptography"})
print(result["report"])



"""
🧠 What’s Happening

Each LLMChain represents an agent’s reasoning block.
SequentialChain executes them in order, passing outputs forward.

You can easily insert:

Memory modules (e.g., ConversationBufferMemory)

Tool calls (e.g., AgentExecutor)

Conditional logic (via RouterChain)

However, the control flow is static — you define the order of execution beforehand.






"""




#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------



"""

🕸️ 3. What is LangGraph?

LangGraph (by LangChain team) is a graph-based orchestration framework built on top of LangChain.

Where LangChain’s SequentialChain is linear, LangGraph defines a dynamic graph of nodes (LLM calls, tools, conditionals)
that can run reactively, handle state, and support multi-agent collaboration.

Key differences:


| Concept                 | LangChain                    | LangGraph                                                 |
| ----------------------- | ---------------------------- | --------------------------------------------------------- |
| **Control Flow**        | Linear / manually sequenced  | Graph-based (dynamic edges, branching, loops)             |
| **State Management**    | Manual or via memory objects | Built-in shared state across nodes                        |
| **Concurrency**         | Limited                      | Supports parallel and asynchronous execution              |
| **Agent Collaboration** | Sequential or nested         | Native multi-agent coordination                           |
| **Use Case**            | Simple pipelines             | Complex reasoning graphs, tool orchestration, async tasks |






"""


#=================================================================
#=======================Lang Graph ===============================
#=================================================================



# pip install langgraph langchain-openai

from langgraph.graph import Graph, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --- Define Node Functions ---

def research_node(state):
    topic = state["topic"]
    result = llm.invoke(f"Search the web for {topic} and summarize 5 key facts.")
    state["research_summary"] = result.content
    return state

def analysis_node(state):
    research_summary = state["research_summary"]
    result = llm.invoke(f"Analyze the following research and extract key insights:\n{research_summary}")
    state["analysis"] = result.content
    return state

def report_node(state):
    analysis = state["analysis"]
    result = llm.invoke(f"Generate a concise technical report based on the following analysis:\n{analysis}")
    state["report"] = result.content
    return state

# --- Build Graph ---
graph = Graph()

graph.add_node("research", research_node)
graph.add_node("analysis", analysis_node)
graph.add_node("report", report_node)

# Edges define workflow
graph.add_edge("research", "analysis")
graph.add_edge("analysis", "report")
graph.add_edge("report", END)

# --- Compile & Run ---
app = graph.compile()

state = {"topic": "impact of quantum computing on cryptography"}
final_state = app.invoke(state)

print(final_state["report"])



"""
🧩 Advantages in LangGraph:

Each node can update shared state (like state["analysis"]).

You can dynamically branch (e.g., skip analysis if already cached).

You can run concurrent nodes (e.g., multiple analysts reviewing data).

Integrates natively with LangChain components (e.g., Agents, Tools, Retrievers).

You could extend this graph to include:

An Evaluation node that checks quality and loops back if unsatisfactory.

A Manager Agent that routes tasks dynamically.



🧠 Summary: LangChain vs LangGraph


| Feature           | LangChain                    | LangGraph                             |
| ----------------- | ---------------------------- | ------------------------------------- |
| Programming Model | Chain-based (sequential)     | Graph-based (nodes & edges)           |
| Flow Control      | Static, linear               | Dynamic, conditional, async           |
| State             | Passed manually              | Shared state across nodes             |
| Ideal for         | Simple pipelines, prototypes | Complex multi-agent, reactive systems |
| Example           | SequentialChain              | Graph().add_node() / add_edge()       |



🚀 When to Use Which

LangChain (classic) → Good for simple, deterministic task flows or when you’re integrating agents/tools linearly.

LangGraph → Best for multi-agent collaboration, stateful orchestration, dynamic branching, or complex reasoning loops.


"""