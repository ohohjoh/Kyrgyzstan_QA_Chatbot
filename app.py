import streamlit as st
import os
from dotenv import load_dotenv
# PyMuPDFLoader는 이제 PDF 원본 로드에 사용되지 않으므로 제거하거나 그대로 두어도 무방합니다.
# 여기서는 사용하지 않으므로 제거했습니다.
from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# .env 파일 로드 (API 키를 환경 변수로 불러오기)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- Streamlit UI 설정 (KBS 디자인 제거) ---
# st.markdown을 통해 삽입했던 커스텀 CSS 스타일 정의 부분을 모두 제거했습니다.
# 따라서 Streamlit의 기본 UI 테마가 적용됩니다.

st.set_page_config(page_title="키르기스스탄 Q&A 챗봇", layout="wide") # 페이지 제목 유지
st.title("🇰🇬 키르기스스탄 Q&A 챗봇") # 메인 제목 유지
st.markdown("PDF 및 마크다운 문서를 기반으로 질문에 답합니다. (출처 포함)") # 설명 문구 유지
st.markdown("---") # 시각적 구분선은 유지

# 세션 상태 초기화 (캐싱 및 대화 기록)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# 세션 상태에 원본 문서 내용을 저장할 딕셔너리 추가
if "full_documents_content" not in st.session_state:
    st.session_state.full_documents_content = {} # Key: source_key, Value: full_content

# --- 문서 로딩 및 처리 함수 (캐싱 적용) ---
@st.cache_resource
def load_and_process_documents(_folder_path="data/processed_md"): # <--- 경로를 전처리된 MD 폴더로 변경!
    all_documents = []
    current_full_documents_content = {}

    # PDF 파일 로드 부분은 이제 제거하고 Markdown 파일 로드만 수행합니다.
    # 이전 주석 처리된 PDF 로드 코드는 여기서도 제거했습니다.

    # 마크다운 파일 로드
    st.info(f"마크다운 파일 로드 중: {_folder_path}/*.md")
    md_loader = DirectoryLoader(
        _folder_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader # UnstructuredMarkdownLoader 사용
    )
    try:
        md_docs = md_loader.load()
        for doc in md_docs:
            source_path = doc.metadata.get("source", "알 수 없는 MD 출처")
            base_name = os.path.basename(source_path)
            
            doc.metadata["source"] = base_name
            doc.metadata["type"] = "Markdown"
            doc.metadata["page"] = "N/A" # 마크다운은 페이지 정보가 없으므로 N/A

            # 원본 문서 내용 저장: 파일 이름만으로 고유 키 생성
            key = f"Markdown_{base_name}"
            current_full_documents_content[key] = doc.page_content
            
        all_documents.extend(md_docs)
        st.success(f"마크다운 파일 {len(md_docs)}개 로드 완료.")
    except Exception as e:
        st.error(f"마크down 파일 로드 중 오류 발생: {e}. '{_folder_path}' 폴더에 마크다운 파일이 있는지 확인해주세요. 'pip install \"unstructured[all-docs]\"' 명령어를 실행했는지 확인하세요.")
        return None # 오류 발생 시 바로 중단

    if not all_documents:
        st.error("로드할 문서가 없습니다. 'data/processed_md' 폴더에 마크다운 파일을 넣어주세요.") # <--- 메시지 변경
        st.stop()
        return None

    # Step 2: 문서 청크 분할 (RecursiveCharacterTextSplitter 사용)
    st.info("문서 청크 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # Markdown 구조를 더 잘 활용하도록 separators 강화
        separators=["\n\n", "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n", ".", " ", ""], 
        length_function=len
    )
    split_docs = text_splitter.split_documents(all_documents)
    st.success(f"총 {len(split_docs)}개의 청크 생성 완료.")

    # Step 3: 벡터 저장소 구축
    st.info("임베딩 및 벡터 저장소 생성 중... (OpenAI API 호출, 시간이 걸릴 수 있습니다)")
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
        st.success("벡터 저장소 생성 완료!")
        
        # vectorstore가 성공적으로 생성되면 원본 문서 내용을 세션 상태에 저장
        st.session_state.full_documents_content = current_full_documents_content
        return vectorstore
    except Exception as e:
        st.error(f"벡터 저장소 생성 중 오류 발생: {e}. OpenAI API 키가 유효한지 확인해주세요.")
        st.stop()
        return None

# --- 메인 애플리케이션 흐름 ---
if st.session_state.vectorstore is None:
    # 문서 로드 경로를 'data/processed_md'로 지정
    st.session_state.vectorstore = load_and_process_documents(_folder_path="data/processed_md")

# vectorstore가 성공적으로 생성되었을 경우에만 챗봇 기능 활성화
if st.session_state.vectorstore:
    @st.cache_resource
    def get_qa_chain(_vectorstore_instance):
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.1, model="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_vectorstore_instance.as_retriever(search_kwargs={"k": 5}), # k값은 필요에 따라 조절
            return_source_documents=True
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
                            if source_type == 'PDF': # 이 부분은 MD만 로드하므로 사실상 호출되지 않지만, 혹시 모를 경우를 위해 유지
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
                                        st.info("원본 내용을 불러올 수 없습니다. 문서 로드 시 문제가 있었거나 해당 부분이 저장되지 않았을 수 있습니다.")
                        else:
                            st.info("참고할 문서가 없습니다.")

                except Exception as e:
                    st.error(f"챗봇 오류: {e}. API 키와 모델 설정을 확인하거나 잠시 후 다시 시도해주세요.")
                    answer = "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다."
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

# vectorstore가 초기화되지 않았을 경우 (로드 실패 시) 메시지 표시
else:
    st.warning("챗봇을 초기화할 수 없습니다. 'data/processed_md' 폴더에 마크다운 파일이 없거나 로드 중 오류가 발생했습니다. 상단의 오류 메시지를 확인해주세요.")