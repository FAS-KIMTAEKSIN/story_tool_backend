from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    EMBEDDING_MODEL = 'text-embedding-3-small'
    GPT_MODEL = "gpt-4o-mini-2024-07-18"
    FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:fasolution::ApD3bgi5"
    MAX_WORKERS = 4
    
    # 파일 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_DB_DIR = os.path.join(BASE_DIR, "test", "vdb")
    TOPIC_MAPPING_PATH = os.path.join(BASE_DIR, "topic_mapping.json")
    
    # 키값 매핑 테이블
    KEY_MAPPING = {
        "genre": "장르",
        "characters": "캐릭터",
        "occupations": "직업/신분",
        "motif": "사건/모티프",
        "location": "공간",
        "emotion": "감정",
        "actions": "행동",
        "relationships": "인물관계",
        "dialogue": "대화형태",
        "keywords": "주제어",
        "theme": "주제문",
    }
    
    # 데이터베이스 설정
    DB_HOST = 'localhost'  # 기본값
    if os.getenv('USE_REMOTE_DB', '').lower() == 'true': # true 일 경우 원격 데이터베이스 사용
        DB_HOST = os.getenv('DB_HOST', 'localhost')
    
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_NAME = os.getenv('DB_NAME') 