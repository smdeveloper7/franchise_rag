# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    VECTOR_DB_PATH: str = "./vector_db/franchise"
    MODEL_NAME: str = "gemini-2.0-flash"
    EMBEDDING_MODEL_NAME: str = "nlpai-lab/KURE-v1"
    EMBEDDING_MODEL_PATH: str = "C:\\Users\\oreo\\.cache\\huggingface\\hub\\models--nlpai-lab--KURE-v1\\snapshots\\d14c8a9423946e268a0c9952fecf3a7aabd73bd9"
    JSON_PATH: str = "./data/test.json"
    DEVICE: str = "cpu"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True  # 대소문자 구분 활성화

# 전역 설정 객체 생성
settings = Settings()