from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from config import settings
import json
import os

# HuggingFace 임베딩 모델 초기화 (KURE-v1)
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_PATH,  # 로컬 경로 사용
        model_kwargs={
            'device': 'cuda'
        }
    )
    print("로컬 임베딩 모델 로드 성공")
except Exception as e:
    print("로컬 임베딩 모델 로드 실패, 온라인 모델을 사용합니다")
    # 실패 시 온라인 모델로 폴백
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )

print("임베딩 모델(KURE-v1) 로딩 완료")

# 벡터 저장소 디렉토리 생성
vector_db_path = settings.VECTOR_DB_PATH
os.makedirs(vector_db_path, exist_ok=True)

# 테스트 JSON 파일 로드
try:
    with open(settings.JSON_PATH, 'r', encoding='utf-8') as file:
        contracts_data = json.load(file)
    print(f"총 {len(contracts_data)} 개의 계약서 데이터를 로드했습니다.")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다: ./test.json")
    contracts_data = []
except json.JSONDecodeError:
    print("JSON 파일 파싱 중 오류가 발생했습니다.")
    contracts_data = []
    exit()

# 문서 객체 생성
documents = []

for contract in contracts_data:
    doc_id = f"{contract['LRN_DTIN_MNNO']}_{contract['CHNK_NO']}"

    metadata = {
        "ID": contract["LRN_DTIN_MNNO"],
        "source": doc_id,
        "brand": contract["JNG_INFO"]["BRAND_NM"],
        "company": contract["JNG_INFO"]["JNGHDQRTRS_CONM_NM"],
        "year": contract["JNG_INFO"]["JNG_BIZ_CRTRA_YR"]
    }
    
    # Content 구성: JSON 객체를 문자열로 변환
    content = [{
        "topic": contract["ATTRB_INFO"]["KORN_UP_ATRB_NM"],
        "sub_topic": contract["ATTRB_INFO"]["KORN_ATTRB_NM"],
        "contents": contract["QL"]["EXTRACTED_SUMMARY_TEXT"] 
    }]
    content_str = json.dumps(content, ensure_ascii=False)  # 리스트를 JSON 문자열로 변환

    # LangChain Document 객체 생성
    doc = Document(page_content=content_str, metadata=metadata)
    documents.append(doc)

print(f"{len(documents)}개의 문서 객체 생성 완료")

# Chroma 벡터 스토어 생성
try:
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="contracts_collection",
        persist_directory=vector_db_path
    )
    print(f"벡터 스토어 생성 완료. 저장 경로: {vector_db_path}")
    
except Exception as e:
    print(f"벡터 스토어 생성 중 오류 발생: {str(e)}")