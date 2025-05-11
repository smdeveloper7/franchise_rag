import os
import json
import time
from pathlib import Path
from config import Settings
import argparse
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
import shutil
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("지식베이스 생성 중입니다...")

# --- JSON_PATH만 외부 인자로 받음 ---
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, required=True, help="테스트 JSON 파일 경로")
parser.add_argument("--device", type=str, default="cuda", help="임베딩 수행 디바이스 (cuda / cpu)")
args = parser.parse_args()


# -----------------------
# 절대경로로 변경
# -----------------------

json_path = Path(args.json_path).resolve()
vector_db_path = Path("./vector_db/franchise").resolve()
settings = Settings(JSON_PATH=str(json_path),VECTOR_DB_PATH=str(vector_db_path),DEVICE=args.device)


if not json_path.exists():
    print(f"❌ JSON 파일이 존재하지 않습니다: {json_path}")
    sys.exit(1)
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        contracts_data = json.load(f)
except Exception as e:
    print(f"❌ JSON 파싱 실패: {e}")
    sys.exit(1)

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
if os.path.exists(vector_db_path):
    shutil.rmtree(vector_db_path)

# 새로 생성
os.makedirs(vector_db_path)

questions = []
documents = []
ids = []
succ_cnt = 0

for idx, contract in enumerate(contracts_data):
    original_text = contract.get("QL", {}).get("EXTRACTED_SUMMARY_TEXT", "").strip()
    if not original_text:
        continue

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
        "source": json_path.name
    }

    doc_id = f"{json_path.name}_{idx}"
    documents.append(Document(page_content=original_text, metadata=metadata))
    ids.append(doc_id)
    succ_cnt += 1

    for qa in contract.get("QL", {}).get("QAs", []):
        questions.append({
            "question": qa["QUESTION"],
            "source_doc": json_path.name,
            "contract_idx": idx
        })


logger.info(f"📄 총 {succ_cnt}개의 문서 처리 완료")
# 질문 저장 경로: JSON 원본과 같은 디렉토리
question_save_path = json_path.parent / f"extract_question_{json_path.name}"

# JSON 파일로 저장
with open(question_save_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

logger.info(f"📝 총 {len(questions)}개의 질문이 추출되어 저장되었습니다.")

# -----------------------
# 벡터 저장소 생성
# -----------------------
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)

try:
    logger.info(f"✅ 벡터 스토어 생성 중... 저장 경로: {settings.VECTOR_DB_PATH}")
    start_time = time.time()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="contracts_collection",
        persist_directory=settings.VECTOR_DB_PATH,
        ids=ids
    )

    elapsed = time.time() - start_time
    logger.info(f"✅ 지식베이스가 생성되었습니다. (⏱️ {elapsed:.2f}초)")
    logger.info(f"📍 저장 위치: {settings.VECTOR_DB_PATH}")
except Exception as e:
    logger.error(f"❌ 벡터 스토어 생성 실패: {e}")
    sys.exit(1)