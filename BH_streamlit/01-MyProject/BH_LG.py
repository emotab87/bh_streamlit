# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("BH_Simple Chatbot")

from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.types import Command


# 체크포인트 저장을 위한 메모리 객체 초기화
memory = MemorySaver()


from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_teddynote.graphs import visualize_graph

def LangGraph_run():
    
    ###### STEP 1. 상태(State) 정의 ######
    class State(TypedDict):
        # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
        messages: Annotated[list, add_messages]
    
    ###### STEP 2. 노드(Node) 정의 ######
    # LLM 정의
    if st.session_state.model_choice == "Anthropic Claude":
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    elif st.session_state.model_choice == "OpenAI ChatGPT":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 챗봇 함수 정의
    def chatbot(state: State):
        # 메시지 호출 및 반환
        return {"messages": [llm.invoke(state["messages"])]}
    
    
    ###### STEP 3. 그래프(Graph) 정의, 노드 추가 ######
    # 그래프 생성
    graph_builder = StateGraph(State)

    # 노드 이름, 함수 혹은 callable 객체를 인자로 받아 노드를 추가
    graph_builder.add_node("chatbot", chatbot)

    ###### STEP 4. 그래프 엣지(Edge) 추가 ######
    # 시작 노드에서 챗봇 노드로의 엣지 추가
    graph_builder.add_edge(START, "chatbot")

    # 그래프에 엣지 추가
    graph_builder.add_edge("chatbot", END)

    ###### STEP 5. 그래프 컴파일(compile) ######
    # 그래프 컴파일
    graph = graph_builder.compile()

    ###### STEP 6. 그래프 시각화 ######
    # 그래프 시각화
    visualize_graph(graph)


    try:
        # Get the mermaid PNG as bytes
        graph_bytes = visualize_graph(graph)

        # Convert the bytes to an image and display it in Streamlit
        st.image(graph_bytes, caption="Chatbot Graph")

    except Exception as e:
        st.error(f"Failed to display graph: {e}")

    if "messages_01" not in st.session_state:
        st.session_state.messages_01 = []

    # Display all previous messages
    for message in st.session_state.messages_01:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get new user input and process it
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages_01.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare the entire chat history to send to the model
        full_conversation = [(msg["role"], msg["content"]) for msg in st.session_state.messages_01]

        # Generate response using the model with the full conversation history
        for event in graph.stream({"messages": full_conversation}):
            for value in event.values():
                # Access the content directly from the AIMessage object
                response = value["messages"][-1].content
                st.session_state.messages_01.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
def main():
    # Sidebar to reset session state
    with st.sidebar:
        if st.button("Reset Session"):
            st.session_state.clear()

    # Page title
    st.title("LangGraph Chatbot Super Basic")

    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "OpenAI ChatGPT"  # Default selection
        st.session_state.model_decided = False  # Track if model has been selected

    # Model selection
    model_choice = st.radio("Choose your model", ["OpenAI ChatGPT", "Anthropic Claude"])   

    # Check if model selection has changed
    if model_choice != st.session_state.model_choice:
        st.session_state.clear()  # Clear current session info
        st.session_state.model_choice = model_choice  # Save new selection to session
        st.session_state.model_decided = True  # Mark that model has been decided

    # Start the chatbot after model selection
    if st.session_state.get('model_decided', False):
        LangGraph_run()

if __name__ == "__main__":
    main()