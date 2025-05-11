# 로깅 설정
import logging
import os

import json
import logging
import os
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiFranchiseService:
    """Chroma 기반 RAG와 Gemini를 활용한 추천 서비스"""
    def __init__(self, api_key: str = None, config: dict = None):
        # 기본 설정 초기화
        self.initial_system_message = (
            "당신은 프랜차이즈 가맹점주를 위한 전문 Q&A 어시스턴트입니다. "
            "당신은 솔트웨어 주식회사가 제공하는 엔터프라이즈 커스터마이징 AI 챗봇 솔루션 'Sapie'입니다. "
            "회사가 제공하는 프랜차이즈 가맹본부 및 가맹점 관련 정보(예: 월매출, 가맹비, 로열티 등)로 임베딩된 테이블 데이터를 사용하여 질문에 답변하세요. "
            "반드시 제공받은 context를 기반으로만 답변해야 하며, 별도로 지식을 추가하거나 재구성해서는 안 됩니다. "
            "만약 질문이 제공된 context와 관련이 없거나, 답변할 정보가 없다면 '문서에 없는 내용입니다. 다시 질문해주세요.'라고 답변하세요. "
            "모든 답변은 친절하고 정확하게 작성하되, 추측은 절대 하지 마세요."
        )
        self.vectorstore_search_k = 5
        self.context_max_length = 8000
        
        genai.configure(api_key=api_key)
        self.vector_db_path = config.get("vector_db_path","")
        self.collection_name = config.get("collection_name","contracts_collection")
        self.model = genai.GenerativeModel(config.get("model_name", "gemini-pro"))
        embedding_model_path = config.get("embedding_model_path","")

        device = config.get("device","cpu")

        
        # 임베딩 모델 초기화 (로컬 모델 사용)
        logger.info(f"로컬 임베딩 모델 로드 중: {embedding_model_path}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_path,  # 로컬 경로 사용
                model_kwargs={
                    'device': device,
                }
            )
            logger.info("로컬 임베딩 모델 로드 성공")
        except Exception as e:
            logger.error(f"로컬 임베딩 모델 로드 실패, 온라인 모델을 사용합니다: {str(e)}")
            # 실패 시 온라인 모델로 폴백
            self.embeddings = HuggingFaceEmbeddings(
                model_name="nlpai-lab/KURE-v1",
                model_kwargs={'device': device}
            )
        
        # Chroma 벡터 스토어 초기화
        self.chroma_vectorstore = self.load_chroma_vectorstore()
    
    def load_chroma_vectorstore(self):
        """LangChain Chroma 벡터스토어 로드"""
        try:
            # 절대 경로로 변환
            absolute_path = os.path.abspath(self.vector_db_path)
            
            # 경로 존재 여부 확인
            if not os.path.exists(absolute_path):
                logger.error(f"벡터 스토어 경로가 존재하지 않습니다: {absolute_path}")
                raise FileNotFoundError(f"벡터 스토어 경로가 존재하지 않습니다: {absolute_path}")
            
            logger.info(f"벡터 스토어 로드 시도: {absolute_path}")
            
            # Chroma 벡터스토어 로드
            vectorstore = Chroma(
                persist_directory=absolute_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # 컬렉션 정보 확인
            collection = vectorstore._collection
            count = collection.count()
            logger.info(f"벡터 스토어 로드 완료: {absolute_path},collection:{collection} 문서 수: {count}")
            
            # 문서가 없는 경우 경고
            if count == 0:
                logger.warning("벡터 스토어에 문서가 없습니다.")
                
            return vectorstore
        except Exception as e:
            logger.error(f"LangChain Chroma 벡터스토어 로드 실패: {str(e)}", exc_info=True)
            raise
        
    def retrieve_context(self, query: str) -> str:
        """Chroma로 문서 검색 및 컨텍스트 생성"""
        try:
            # 기본적인 similarity search 사용
            search_results = self.chroma_vectorstore.similarity_search(
                query=query,
                k=self.vectorstore_search_k
            )
            
            # 검색된 문서로 컨텍스트 구성
            context = ""
            total_length = 0
            
            for i, doc in enumerate(search_results, 1):
                # JSON 문자열을 파싱
                try:
                    content_json = json.loads(doc.page_content)
                    # JSON 구조에 맞게 컨텐츠 추출
                    if isinstance(content_json, list) and len(content_json) > 0:
                        doc_text = ""
                        for item in content_json:
                            topic = item.get("topic", "")
                            sub_topic = item.get("sub_topic", "")
                            contents = item.get("contents", "")
                            doc_text += f"주제: {topic}\n소주제: {sub_topic}\n내용: {contents}\n\n"
                    else:
                        doc_text = str(content_json)
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 원본 텍스트 사용
                    doc_text = doc.page_content
                
                doc_length = len(doc_text)
                if total_length + doc_length > self.context_max_length:
                    continue
                
                # 메타데이터가 있으면 출처 정보 추가
                metadata_str = ""
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata_str = " | " + " | ".join([f"{k}: {v}" for k, v in doc.metadata.items() if k != "text"])
                
                context += f"[문서 {i}]{metadata_str}\n{doc_text}\n\n"
                total_length += doc_length

            logger.info(f"Chroma 검색 완료: {len(search_results)}개 문서, 컨텍스트 길이: {total_length}")
            return context
        except Exception as e:
            logger.error(f"Chroma 검색 실패: {str(e)}")
            return ""

    def answer_question(self, query: str) -> str:
        """사용자 질문에 RAG를 통해 답변"""
        try:
            # 컨텍스트 검색
            context = self.retrieve_context(query)
            
            if not context:
                return "검색 결과가 없습니다. 다른 질문을 해주세요."
            
            # Gemini 프롬프트 구성
            prompt = f"""
            {self.initial_system_message}

            컨텍스트:
            {context}

            사용자 질문: {query}

            답변:
            """
            
            # Gemini 모델로 답변 생성
            response = self.model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"질문 답변 실패: {str(e)}")
            return f"죄송합니다, 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def add_documents(self, documents):
        """벡터스토어에 새 문서 추가"""
        try:
            self.chroma_vectorstore.add_documents(documents)
            logger.info(f"{len(documents)}개 문서가 벡터스토어에 추가되었습니다.")
            return True
        except Exception as e:
            logger.error(f"문서 추가 실패: {str(e)}")
            return False