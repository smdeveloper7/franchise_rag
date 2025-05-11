# 🧠 Franchise RAG System

프랜차이즈 정보공개서 기반 QA 시스템입니다.  
few-shot QA 데이터셋을 벡터화하여 RAG 형태로 질의응답을 수행합니다.

---

## 📦 실행 순서

```bash
git clone https://github.com/smdeveloper7/franchise_rag
```

### 1. 가상환경 설정

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Few-shot QA 벡터 DB 생성

```bash
python3 qa_knowledge_base.py
```

### 3. 추론 실행 (테스트 QA필요, 현재 임의로 5개만들었음 1059017501.json )
```bash
bash franchise_RAG.sh ./data/test/1059017501.json
```

### 4. 결과 터미널 표시시
✅ 결과 저장 완료: /home/sm7540/workspace/franchise_rag/data/result/test_data.json
