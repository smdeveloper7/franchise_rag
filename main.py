from franchise import GeminiFranchiseService
from config import settings
if __name__ == "__main__":
    # Gemini API 키 설정
    GEMINI_API_KEY = settings.GEMINI_API_KEY
    
    # 서비스 초기화
    rag_service = GeminiFranchiseService(
        api_key=GEMINI_API_KEY,
        config={
            "vector_db_path": settings.VECTOR_DB_PATH,
            "model_name": settings.MODEL_NAME,
            "embedding_model_path": settings.EMBEDDING_MODEL_PATH
        }
    )
    
    # 테스트 질문
    test_questions = [
        "서영에프앤비의 주소는 어디인가요?",
        "(주)서영에프앤비에서 운영하는 브랜드는?",
        # "피자헛의 창업 비용은 얼마인가요?" # 데이터에 없는 질문
    ]
    
    # 질문 답변 테스트
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 질문 {i}: {question} ---")
        answer = rag_service.answer_question(question)
        print(f"답변: {answer}")