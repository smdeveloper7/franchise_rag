import json
import argparse
from pathlib import Path
from config import Settings
from fewshot_franchise import GeminiFewShotFranchiseService

# ---------------------
# 인자 처리
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, required=True, help="테스트 JSON 파일 경로")
parser.add_argument("--device", type=str, default="cuda", help="임베딩 수행 디바이스 (cuda / cpu)")
args = parser.parse_args()

# ---------------------
# 질문 파일 경로 구성
# ---------------------
json_path = Path(args.json_path).resolve()
questions_path = json_path.parent / f"extract_question_{json_path.name}"

# ---------------------
# Settings + 서비스 초기화
# ---------------------
settings = Settings(DEVICE=args.device)

rag_service = GeminiFewShotFranchiseService(
    api_key=settings.GEMINI_API_KEY,
    config={
        "vector_db_path": settings.VECTOR_DB_PATH,
        "model_name": settings.MODEL_NAME,
        "embedding_model_path": settings.EMBEDDING_MODEL_PATH,
        "device":settings.DEVICE,
        "vectorstore_search_k":1
    }
)


# ---------------------
# 질문 파일 로딩
# ---------------------
if not questions_path.exists():
    raise FileNotFoundError(f"❌ 질문 파일이 존재하지 않습니다: {questions_path}")

with open(questions_path, "r", encoding="utf-8") as f:
    questions_data = json.load(f)


results = []
for i, q_item in enumerate(questions_data[:5]):
    query = q_item["question"]
    result = rag_service.inference(query)
    results.append(result)


output_path = Path("./data/result/test_data.json").resolve()
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✅ 결과 저장 완료: {output_path}")

# ---------------------
# 질문 순회 추론
# ---------------------
# for i, q_item in enumerate(questions_data):
#     question = q_item["question"]
#     print(f"\n🔎 [{i+1}/{len(questions_data)}] 질문: {question}")

#     try:
#         response = rag_service.run(question)
#         print(f"💬 응답:\n{response}")
#     except Exception as e:
#         print(f"❌ 에러 발생: {e}")
