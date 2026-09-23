from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Mock(Runnable):
    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content='{"agent": "CONVERSATION_AGENT", "urgency": "ROUTINE", "reasoning": "ok", "confidence": 0.95}')

p = ChatPromptTemplate.from_messages([("human", "{input}")])
c = p | Mock() | JsonOutputParser()
res = c.invoke({"input": "hello"})
print("Success result:", res)
