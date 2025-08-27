from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

class Agent:
    def __init__(self, system_msg, tools):
        self.system_msg = system_msg
        self.tools = tools

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print('BT - Agent called...')
        print("---CALL AGENT---")
        messages = state["messages"]
        print('BT - Agent receiving messages: ', messages)

        # Insert system message once at the beginning
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [self.system_msg] + messages

        model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
        print("Binding tools:", [tool.name for tool in self.tools])
        model = model.bind_tools(self.tools)
        response = model.invoke(messages)
        print('BT - Agent receiving message back from LLM: ', response)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
