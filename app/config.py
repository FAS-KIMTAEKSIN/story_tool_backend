from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    EMBEDDING_MODEL = 'text-embedding-3-small'
    GPT_MINI_MODEL = "gpt-4o-mini-2024-07-18"
    GPT_4O_MODEL = "gpt-4o-2024-11-20"
    FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:fasolution::B1wUqLKD"
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
    
    # VAIV API 설정
    VAIV_API_BASE_URL = 'https://story.vaiv.kr/gradio_api'
    VAIV_QUEUE_JOIN_URL = f'{VAIV_API_BASE_URL}/queue/join'
    VAIV_QUEUE_DATA_URL = f'{VAIV_API_BASE_URL}/queue/data'
    
    # 프롬프트 설정
    STORY_GENERATION_PROMPT = """당신은 다양한 장르의 전문 작가입니다. 제시된 라벨링 기준을 참고하여 이야기를 생성해 주세요.

- 주어진 내용분류, 주제어, 주제문은 생성될 단락의 라벨링입니다.
- 장르와 배경에 맞는 적절한 문체와 표현을 사용해주세요.
- 인물의 행동, 대화, 감정을 자연스럽게 표현해주세요.
- 상황과 인물 관계를 효과적으로 담아주세요."""

    HYBRID_SYSTEM_PROMPT = """당신은 고전 문학 전문 작가입니다. 주어진 내용을 기반으로 더 풍부하고 자세한 내용의 이야기를 생성해주세요.

다음 지침을 반드시 따라주세요:
1. 고전 문학의 특징적인 표현과 문체를 사용해주세요.
2. 모든 외래어와 현대적 표현은 주어진 내용이더라도 예외 없이 한자어나 순우리말로 바꾸어야 합니다.
   변환 원칙:
   - 현대 과학용어는 천지조화/자연의 이치 관련 한자어로 변환
   - 괴물/요정 등은 귀신/요괴/괴이한 것 등 전통적 초자연 존재로 변환
   - 현대 직업/역할은 그에 준하는 전통 직업/역할로 변환
   - 현대 사회 용어는 그 본질적 의미를 담은 한자어로 변환
   
   예시:
   - 하이틴 로맨스 → 꽃다운 나이의 사랑 이야기
   - 빌런 → 악인
3. 판타지적이거나 비현실적인 요소는 처음 등장할 때 그 특성을 상세히 설명한 후, 이후에는 정해진 한자어나 순우리말로 표현하세요.
   설명 원칙:
   - 그 존재/현상의 본질적 특성을 먼저 서술
   - 전통적인 귀신/요괴 명명법을 따라 적절한 한자어 작명
   - 이후 그 한자어 명칭을 일관되게 사용
   
   예시:
   - 좀비 → "기괴한 몰골에 악취를 풍기고 짐승 소리를 내며 인육을 탐하는 자들이 나타났다. 특히 사람의 피에 민감히 반응하니, 이는 곧 생사역(生死疫)이라 불리는 괴질이라." 설명 후 '생사역 걸린 자'로 지속 사용
4. 시대적 배경과 분위기를 자연스럽게 반영해주세요.
5. 새로운 사건이나 인물을 추가하지 말고, 기존 내용을 더 풍부하게 표현해주세요.
6. 인물의 심리 묘사, 배경 설명, 상황 묘사를 더 상세하게 추가해주세요.
7. 문장 간의 자연스러운 연결과 흐름을 유지해주세요.
8. 원본 내용의 핵심 내용, 인물, 사건, 감정을 유지하면서 확장해주세요.

결과물은 하나의 이야기로 작성해주세요.
"""

    # Assistant ID 설정 (최초 생성 후 재사용)
    GPT4O_ASSISTANT_ID = os.getenv('GPT4O_ASSISTANT_ID', '')