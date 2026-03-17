"""
Agentic AI Workflow — Multi-step Document Agent
LangChain Agents + ReAct framework for autonomous document retrieval,
analysis, and summarization with LLM guardrails and output validation.
"""

import os
from typing import Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, field_validator
import re

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")


# --- Output validation model ---
class AgentOutput(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

    @field_validator("answer")
    @classmethod
    def no_pii(cls, v: str) -> str:
        """Strip potential PII patterns from LLM output."""
        v = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", v)
        v = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL REDACTED]", v)
        return v

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return v


# --- Document store ---
def build_faiss_index(file_paths: list[str]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = []
    for path in file_paths:
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        docs.extend(splitter.split_documents(loader.load()))
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(FAISS_INDEX_PATH)
    return store


def load_faiss_index() -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


# --- Tools ---
def make_retrieval_tool(store: FAISS) -> Tool:
    def retrieve(query: str) -> str:
        docs = store.similarity_search(query, k=4)
        return "\n\n".join(d.page_content for d in docs)

    return Tool(
        name="DocumentRetrieval",
        func=retrieve,
        description="Retrieve relevant passages from the document knowledge base given a query.",
    )


def make_summarization_tool(llm: ChatOpenAI) -> Tool:
    summarize_chain = (
        PromptTemplate.from_template("Summarize the following text concisely:\n\n{text}")
        | llm
        | StrOutputParser()
    )

    def summarize(text: str) -> str:
        return summarize_chain.invoke({"text": text})

    return Tool(
        name="Summarizer",
        func=summarize,
        description="Summarize a long passage of text into a concise paragraph.",
    )


# --- Agent ---
REACT_PROMPT = PromptTemplate.from_template("""You are an expert document analyst. Use the tools available to
answer the question. Always retrieve relevant context before answering.

Tools: {tools}
Tool names: {tool_names}

Question: {input}
{agent_scratchpad}""")


def build_agent(store: FAISS) -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    tools = [make_retrieval_tool(store), make_summarization_tool(llm)]
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6, handle_parsing_errors=True)


def run_agent(question: str) -> AgentOutput:
    store = load_faiss_index()
    executor = build_agent(store)
    raw = executor.invoke({"input": question})
    return AgentOutput(answer=raw["output"], confidence=0.9, sources=[])


if __name__ == "__main__":
    result = run_agent("What are the key findings in the financial report?")
    print(result.model_dump_json(indent=2))
