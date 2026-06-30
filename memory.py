from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

SUMMARY_LLM = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

SUMMARY_TEMPLATE = """Summarize the following conversation concisely, capturing the key entities (people, teams), facts, and the last question asked.

Conversation:
{conversation}

Concise summary:"""


class ConversationMemory:
    def __init__(self, max_buffer_turns=4):
        self.max_buffer_turns = max_buffer_turns
        self.turns = []
        self.summary = ""

    def add_turn(self, user_message, assistant_message):
        self.turns.append({"user": user_message, "assistant": assistant_message})
        if len(self.turns) > self.max_buffer_turns * 2:
            self._roll_summary()

    def _roll_summary(self):
        old_turns = self.turns[:len(self.turns) - self.max_buffer_turns]
        conversation_text = "\n".join(
            f"User: {t['user']}\nAssistant: {t['assistant']}" for t in old_turns
        )
        if self.summary:
            conversation_text = f"Previous summary: {self.summary}\n\n{conversation_text}"
        prompt = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)
        chain = prompt | SUMMARY_LLM | StrOutputParser()
        self.summary = chain.invoke({"conversation": conversation_text})
        self.turns = self.turns[len(self.turns) - self.max_buffer_turns:]

    def context(self):
        buffer = "\n".join(
            f"User: {t['user']}\nAssistant: {t['assistant']}" for t in self.turns
        )
        if self.summary:
            return f"Conversation summary (earlier turns):\n{self.summary}\n\nRecent conversation:\n{buffer}"
        return f"Conversation so far:\n{buffer}" if buffer else "No prior conversation."

    def to_history(self):
        return list(self.turns)

    def clear(self):
        self.turns = []
        self.summary = ""
