from langchain_aws import BedrockEmbeddings
from langchain.docstore.document import Document

from langchain_chroma import Chroma
import boto3
import json
bedrock = boto3.client(
    'bedrock-runtime',
    region_name='us-west-2',  
)

embedding_function = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v1"
)

# JSON 파일 로드
try:
    with open('./test.json', 'r', encoding='utf-8') as file:
        contracts_data = json.load(file)
    print(f"총 {len(contracts_data)} 개의 계약서 데이터를 로드했습니다.")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다: /data/merged_contracts.json")
    contracts_data = []
except json.JSONDecodeError:
    print("JSON 파일 파싱 중 오류가 발생했습니다.")
    contracts_data = []
    exit()

# 문서 객체 생성
ids = []
documents = []
metadatas = []

for contract in contracts_data:
    doc_id = f"{contract['LRN_DTIN_ID']}_{contract['chunk']}"

    metadata = {"ID": contract["LRN_DTIN_ID"]}
    
    # Content 구성: JSON 객체를 문자열로 변환
    content = [{
        "topic": contract["ATTRB_INFO"]["KORN_UP_ATRB_NM"],
        "sub_topic": contract["ATTRB_INFO"]["KORN_ATTRB_NM"],
        "contents": contract["QL"]["extracted_summary_text"]
    }]
    content_str = json.dumps(content, ensure_ascii=False)  # 리스트를 JSON 문자열로 변환

    # LangChain Document 객체 생성
    doc = Document(page_content=content_str, metadata=metadata)
    documents.append(doc)
    
# Chroma 클라이언트 초기화 (로컬 저장소 사용)
vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name="contracts_collection",
        persist_directory="./chroma_db"  # 로컬 저장 경로
    )