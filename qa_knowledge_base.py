from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from config import Settings
from pathlib import Path
import json
import os
import time

## 사전 train 데이터셋을 활용한 지식베이스 생성
settings = Settings(JSON_PATH="./data/train",VECTOR_DB_PATH="./vector_db/qa_knowledge_base") ## json 경로 설정

# HuggingFace 임베딩 모델 초기화
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda'}
    )
    print("로컬 임베딩 모델 로드 성공")
except Exception:
    print("로컬 임베딩 모델 로드 실패, 온라인 모델을 사용합니다")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )

print(f"임베딩 모델 {settings.EMBEDDING_MODEL_NAME} 로딩 완료")

# 벡터 저장소 디렉토리 생성
vector_db_path = settings.VECTOR_DB_PATH
os.makedirs(vector_db_path, exist_ok=True)

# JSON 데이터 로드


# 문서 객체 생성
documents = []
ids = []

succ_cnt = 0 # 성공 갯수

# JSON 경로를 Path 객체로 변환
json_dir = Path(settings.JSON_PATH)
# .json 파일 전체 경로 리스트
json_files = list(json_dir.glob("*.json"))  

for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            contracts_data = json.load(file)
        # print(f"총 {len(contracts_data)} 개의 계약서 데이터를 로드했습니다.")
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {json_file}")
        contracts_data = []
        exit()
    except json.JSONDecodeError:
        print(f"{json_file} 파일 파싱 중 오류가 발생했습니다.")
        contracts_data = []
        exit()

    for contract_idx, contract in enumerate(contracts_data):
        summary_text = contract.get("QL", {}).get("ABSTRACTED_SUMMARY_TEXT", "").strip()
        qas = contract.get("QL", {}).get("QAs", [])
        # 메타데이터 구성
        metadata = {
            "CHNK_NO": contract["CHNK_NO"],
            "SMRT_CHNK_NO": contract["SMRT_CHNK_NO"],
            "JNG_BIZ_CRTRA_YR": contract["JNG_INFO"]["JNG_BIZ_CRTRA_YR"],
            "JNGHDQRTRS_CONM_NM": contract["JNG_INFO"]["JNGHDQRTRS_CONM_NM"],
            "BRAND_NM": contract["JNG_INFO"]["BRAND_NM"],
            "JNG_IFRMP_SN": contract["JNG_INFO"]["JNG_IFRMP_SN"],
            "ATTRB_MNNO": contract["ATTRB_INFO"]["ATTRB_MNNO"],
            "KORN_ATTRB_NM": contract["ATTRB_INFO"]["KORN_ATTRB_NM"],
            "UP_ATTRB_MNNO": contract["ATTRB_INFO"]["UP_ATTRB_MNNO"],
            "KORN_UP_ATRB_NM": contract["ATTRB_INFO"]["KORN_UP_ATRB_NM"],
            "source": f"{json_file.name}",
            "QAs": json.dumps(qas, ensure_ascii=False)  # 문자열로 저장  # QA 전체 포함
        }
        
        doc_id = f"{json_file.name}_{contract_idx}"
        documents.append(Document(page_content=summary_text, metadata=metadata))
        ids.append(doc_id)
        succ_cnt+=1

print(f"📄 총 {succ_cnt}개의 문서 처리 완료")

# Chroma 벡터 스토어 생성
try:
    print(f"✅ 벡터 스토어 생성 중... 저장 경로: {vector_db_path}")
    start_time = time.time()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="contracts_qa_collection",
        persist_directory=vector_db_path,
        ids=ids
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"✅ 벡터 스토어 생성 완료. 소요 시간: {elapsed_time:.2f}초")
    print(f"📍 저장 경로: {vector_db_path}")
except Exception as e:
    print(f"❌ 벡터 스토어 생성 중 오류 발생: {str(e)}")
