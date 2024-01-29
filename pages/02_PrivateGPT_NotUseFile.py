from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import streamlit as st

# page_title
st.set_page_config(
    page_title="PrivateGPT_NotUseFile",
    page_icon="📃",
)


# 채팅 콜백 핸들러 클래스
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# Ollama 모델을 사용하여 채팅 기능 구현
# Ollama 서버(로컬, 서버 모두)와 통신이 안된다면, 네트워크 오류 발생
llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


# 사용자와 AI의 메시지를 저장하는 함수
def save_message(message, role):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    st.session_state["messages"].append({"message": message, "role": role})


# 메시지 전송 및 화면에 표시 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# 이전 대화 내용을 화면에 표시하는 함수
def paint_history():
    for message in st.session_state.get("messages", []):
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


# 채팅 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_template("""{question}""")

# 웹 애플리케이션 제목 및 설명
st.title("PrivateGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to interact with an AI!
"""
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 대화 시작
send_message("I'm ready! Ask away!", "ai", save=False)
paint_history()
message = st.chat_input("What's your question?")
if message:
    send_message(message, "human")
    chain = (
        {
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    with st.chat_message("ai"):
        chain.invoke(message)
