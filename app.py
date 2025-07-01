# app.py

# Streamlit Cloud 환경에서만 pysqlite3를 사용하도록 조건부 임포트
# 로컬 개발 환경에서는 오류를 방지하고, 기본 sqlite3를 사용합니다.
import os
if os.environ.get('STREAMLIT_SERVER_PORT') or os.environ.get('IS_STREAMLIT_CLOUD'):
    try:
        import pysqlite3
        import sys
        sys.modules["sqlite3"] = sys.modules["pysqlite3"]
    except ImportError:
        # Streamlit Cloud 환경이지만 pysqlite3가 설치되지 않은 경우
        print("Warning: pysqlite3 not found, falling back to default sqlite3.")
        pass

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI Embeddings와 Chat 모델은 각각 아래에서 가져옵니다.
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# from langchain.schema import SystemMessage # 이 줄은 더 이상 필요 없으므로 제거했습니다.

# .env 파일 로드 (API 키를 환경 변수로 불러오기)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="키르기스스탄 Q&A 챗봇", layout="wide")
st.title("🇰🇬 키르기스스탄 Q&A 챗봇")
st.markdown("마크다운 문서를 기반으로 질문에 답합니다. (출처 포함)")
st.markdown("---")

# 세션 상태 초기화 (캐싱 및 대화 기록)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# 세션 상태에 원본 문서 내용을 저장할 딕셔너리 추가
if "full_documents_content" not in st.session_state:
    st.session_state.full_documents_content = {} # Key: source_key, Value: full_content

# --- 문서 로딩 및 처리 함수 (캐싱 적용) ---
@st.cache_resource(show_spinner=False) # 스피너 메시지를 직접 제어하기 위해 show_spinner=False
def load_and_process_documents(_folder_path="data/processed_md"):
    all_documents = []
    current_full_documents_content = {}

    # 마크다운 파일 로드
    st.info(f"✨ 마크다운 파일 로드 중: `{_folder_path}/*.md`")
    md_loader = DirectoryLoader(
        _folder_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )
    try:
        md_docs = md_loader.load()
        if not md_docs:
            st.error("⚠️ 로드할 마크다운 문서가 없습니다. 'data/processed_md' 폴더에 파일을 넣어주세요.")
            st.stop() # 문서를 찾지 못하면 앱 실행 중단
            return None

        for doc in md_docs:
            source_path = doc.metadata.get("source", "알 수 없는 MD 출처")
            base_name = os.path.basename(source_path)
            
            doc.metadata["source"] = base_name
            doc.metadata["type"] = "Markdown"
            doc.metadata["page"] = "N/A" # 마크다운은 페이지 정보가 없으므로 N/A

            key = f"Markdown_{base_name}"
            current_full_documents_content[key] = doc.page_content
            
        all_documents.extend(md_docs)
        st.success(f"✅ 마크다운 파일 {len(md_docs)}개 로드 완료.")
    except Exception as e:
        st.error(f"❌ 마크다운 파일 로드 중 오류 발생: {e}")
        st.error(f"'{_folder_path}' 폴더에 마크다운 파일이 있는지, 그리고 'pip install \"unstructured[all-docs]\"'를 실행했는지 확인해주세요.")
        st.stop()
        return None

    # Step 2: 문서 청크 분할 (RecursiveCharacterTextSplitter 사용)
    st.info("✂️ 문서 청크 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n", ".", " ", ""], 
        length_function=len
    )
    split_docs = text_splitter.split_documents(all_documents)
    st.success(f"✅ 총 {len(split_docs)}개의 청크 생성 완료.")

    # Step 3: 벡터 저장소 구축
    st.info("🧠 임베딩 및 벡터 저장소 생성 중... (OpenAI API 호출, 시간이 걸릴 수 있습니다)")
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
        st.success("✅ 벡터 저장소 생성 완료!")
        
        # vectorstore가 성공적으로 생성되면 원본 문서 내용을 세션 상태에 저장
        st.session_state.full_documents_content = current_full_documents_content
        
        return vectorstore
    except Exception as e:
        st.error(f"❌ 벡터 저장소 생성 중 오류 발생: {e}. OpenAI API 키가 유효한지, 충분한 잔액이 있는지 확인해주세요.")
        st.stop()
        return None

# --- 메인 애플리케이션 흐름 ---
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_and_process_documents(_folder_path="data/processed_md")

# vectorstore가 성공적으로 생성되었을 경우에만 챗봇 기능 활성화
if st.session_state.vectorstore:
    @st.cache_resource(show_spinner=False)
    def get_qa_chain(_vectorstore_instance):
        # 1) 시스템 메시지 템플릿 정의
        system_prompt = """
        당신은 KBS(한국방송공사)와 방송기술인협회로 구성된 컨소시엄의 ODA(공적개발원조) 및 KOICA(한국국제협력단) 사업 전문 자문가입니다.
        컨소시엄은 현재 키르기스스탄 정부가 공모하는 ODA/KOICA 사업 수주를 목표로 하고 있습니다.
        당신은 제공된 문서(마크다운 파일)를 기반으로 컨소시엄이 성공적으로 사업을 수주하고 진행할 수 있도록 전략적이고 실질적인 조언과 정보를 제공해야 합니다.

        다음 지침을 엄격히 준수하여 답변해주세요:
        1.  **주요 역할:** ODA 및 KOICA 사업 전문가로서 컨소시엄(KBS 및 방송기술인협회)의 입장을 대변하며, 사업 수주 및 실행에 필요한 모든 정보를 제공합니다.
        2.  **관점 유지:** 특별한 요구가 없는 한, 항상 KBS와 방송기술인협회 컨소시엄의 이익과 관점에서 답변을 구성합니다. 사업의 기회, 컨소시엄의 강점, 잠재적 위험 및 해결 방안 등을 이들의 입장에서 분석하고 제시합니다.
        3.  **답변 언어:** 모든 답변은 반드시 명확하고 전문적인 **한국어**로 제공합니다.
        4.  **정보의 근거:** 답변의 모든 내용은 오직 당신에게 제공된 문서의 내용만을 바탕으로 합니다. 문서에 없는 정보는 절대로 추측하거나 지어내지 마세요.
        5.  **질문 맥락:** 사용자의 질문이 ODA/KOICA 사업 공모, 키르기스스탄 정부와의 협력, 사업 제안서 작성, 기술 지원, 컨소시엄 운영 등과 관련된 것임을 항상 인지하고 답변합니다.
        6.  **답변 형식:**
            * 핵심 내용을 먼저 간결하게 요약하고, 필요한 경우 상세한 설명을 덧붙입니다.
            * 전문 용어는 명확하게 설명해주되, 컨소시엄 관계자들이 이해하기 쉽도록 풀어서 설명합니다.
        7.  **출처 명시:** 답변의 마지막에는 항상 "답변 참고 자료" 섹션에 제시되는 출처 문서의 제목을 간결하게 언급하여 답변의 신뢰도를 높입니다.
        8.  **정보 부족 시 대처:** 만약 질문에 대한 정보가 제공된 문서 내에 충분하지 않거나 찾을 수 없다면,
        "죄송합니다. 현재 문서에서는 해당 정보를 찾을 수 없습니다. 더 자세한 정보가 필요하시면 관련 문서를 추가해주시면 도움을 드릴 수 있습니다."
        라고 답변합니다.
        """

        # 2) 사용자 질문 부분 템플릿 정의
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

        # 3) LLM 인스턴스 생성
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0.1,
            model_name="gpt-3.5-turbo"
        )

        # 4) RetrievalQA 체인 구성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vectorstore_instance.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": chat_prompt}
        )
        return qa_chain

    qa_chain = get_qa_chain(st.session_state.vectorstore)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("문서에 대해 질문해주세요.")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    result = qa_chain({"query": prompt})
                    answer = result["result"]
                    source_documents = result["source_documents"]

                    st.markdown(answer)

                    if source_documents:
                        st.markdown("---")
                        st.markdown("### 📚 답변 참고 자료")
                        unique_sources_for_display = {} 

                        for doc in source_documents:
                            source_file_name = doc.metadata.get('source', '알 수 없는 출처')
                            source_type = doc.metadata.get('type', '문서 유형 없음')
                            page_info = doc.metadata.get('page', 'N/A')
                            
                            source_key_for_content = ""
                            if source_type == 'PDF': # 현재는 MD만 로드하므로 사실상 호출되지 않지만, 유지
                                source_key_for_content = f"PDF_{source_file_name}_page_{page_info}"
                            elif source_type == 'Markdown':
                                source_key_for_content = f"Markdown_{source_file_name}"
                            
                            unique_sources_for_display[source_key_for_content] = {
                                "display_name": f"**{source_type}:** `{source_file_name}`" + (f" (페이지: {page_info})" if source_type == 'PDF' and page_info != 'N/A' else ""),
                                "content_key": source_key_for_content
                            }
                        
                        sorted_sources = sorted(unique_sources_for_display.values(), key=lambda x: x["display_name"])

                        if sorted_sources:
                            for source_info in sorted_sources:
                                with st.expander(source_info["display_name"]):
                                    content_key = source_info["content_key"]
                                    if content_key in st.session_state.full_documents_content:
                                        st.text_area(
                                            label="관련 내용", 
                                            value=st.session_state.full_documents_content[content_key],
                                            height=200, 
                                            disabled=True, 
                                            key=f"content_area_{content_key}" 
                                        )
                                    else:
                                        st.info("⚠️ 원본 내용을 불러올 수 없습니다. 문서 로드 시 문제가 있었거나 해당 부분이 저장되지 않았을 수 있습니다.")
                        else:
                            st.info("참고할 문서가 없습니다.")

                except Exception as e:
                    st.error(f"❌ 챗봇 오류: {e}. API 키와 모델 설정을 확인하거나 잠시 후 다시 시도해주세요.")
                    answer = "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다."
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

# vectorstore가 초기화되지 않았을 경우 (로드 실패 시) 메시지 표시
else:
    st.warning("⚠️ 챗봇을 초기화할 수 없습니다. 상단의 오류 메시지를 확인해주세요.")