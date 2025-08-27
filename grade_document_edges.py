from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class GradeDocumentEdges:
    """
    Class to encapsulate the logic for grading document relevance.
    """
    def __init__(self):
        self.model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
        self.prompt = PromptTemplate(
            template="""
You are a strict grader that evaluates whether a retrieved document is relevant to a user's question.

Your task is to return a binary score:
- "yes" if the document clearly contains keywords, phrases, or exact instructions that directly answer or explain the user’s question.
- "no" if the document does not mention any of the keywords from the question or does not directly relate to the user's intent.

Do not infer or assume relevance based on vague similarities. Only return "yes" if the answer is explicitly addressed in the document.

---

User Question:
{question}

Retrieved Document:
{context}

Does the document directly answer or relate to the question? Reply with "yes" or "no".
""",
            input_variables=["context", "question"],
        )

    def grade_documents(self, state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.
        Uses a quick keyword match as a pre-check, and then validates via LLM with strict prompt.
        """
        print('BT - grade_document called...')
        print("---CHECK RELEVANCE---")

        # Extract user question and retrieved docs
        messages = state["messages"]
        question = messages[0].content
        docs = messages[-1].content

        # --- Quick keyword overlap check ---
        def contains_keywords(question, docs):
            question_keywords = set(question.lower().split())
            doc_text = docs.lower()
            return any(keyword in doc_text for keyword in question_keywords)

        if not contains_keywords(question, docs):
            print("---PRE-CHECK FAILED: No keyword match---")
            return "rewrite"

        # --- If keyword check passes, do strict LLM grading ---
        class grade(BaseModel):
            """Binary score for relevance check."""
            binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

        llm_with_tool = self.model.with_structured_output(grade)
        chain = self.prompt | llm_with_tool
        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score.strip().lower()

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"
