from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class RewriteAgent:
    def __init__(self):
        pass

    def rewrite(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print('BT - rewrite called...')
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f"""
Look at the input and try to reason about the underlying semantic intent / meaning.\n\nHere is the initial question:\n-------\n{question}\n-------\nFormulate an improved question: """,
            )
        ]

        # Grader
        model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
        response = model.invoke(msg)
        return {"messages": [response]}
