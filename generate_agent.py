from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class GenerateAgent:
    def __init__(self):
        pass

    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
             dict: The updated state with re-phrased question
        """
        print('BT - generate called...')
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content

        prompt = PromptTemplate(
            template="""You are a helpful assistant for question-answering tasks. Use the following instructions to respond accurately and reliably:
                1. First, use only the retrieved internal context below to answer the question. If not enough information, use the Tavily web search tool to find an appropriate answer.
                2. Do **not** mix information from different sources. Clearly state whether your answer is based on internal documents or external web results.
                3. Clearly distinguish between the Multitech Dot and the Conduit/Multitech Gateway — do not confuse or substitute one for the other under any circumstances.
                4. Provide a complete, step-by-step instruction when applicable, and include any screenshots, links, or examples exactly as they appear in the context or search results.
                5. Always cite the source of the information (e.g., document name or URL).
                6. If no relevant information is found from either internal documents or web search, respond with: \"I don’t know.\"
                Question: {question}
                Context: {context}
                Answer:""",
            input_variables=["context", "question"],
        )

        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
        rag_chain = prompt | llm
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
