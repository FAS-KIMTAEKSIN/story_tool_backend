from openai import OpenAI
from app.config import Config
from app.utils.database import Database
from app.utils.helpers import generate_session_hash
import json
import traceback
import time
import requests
import logging
from typing import Optional, Dict, Any, Generator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 싱글톤
openai_client = OpenAI()

class ThreadManager:
    """Thread 관리를 담당하는 클래스"""
    
    @staticmethod
    def create_thread(user_id: str, title: str, openai_thread_id: str) -> int:
        """새로운 thread 생성"""
        with Database() as db:
            db.connection.start_transaction()
            try:
                db.execute(
                    """INSERT INTO threads (user_id, title, thread_id) 
                       VALUES (%s, %s, %s)""",
                    (user_id, title, openai_thread_id)
                )
                thread_id = db.lastrowid
                db.connection.commit()
                logger.info(f"Created new thread: {thread_id}", extra={"thread_id": thread_id})
                return thread_id
            except Exception as e:
                db.connection.rollback()
                logger.error(f"Failed to create thread: {str(e)}", exc_info=True)
                raise

    @staticmethod
    def update_thread_title(thread_id: int, title: str, only_if_new: bool = False) -> bool:
        """thread 제목 업데이트"""
        if only_if_new:
            with Database() as db:
                db.execute(
                    """SELECT id FROM threads WHERE id = %s AND title = '새 이야기'""",
                    (thread_id,)
                )
                if not db.fetchone():
                    return False

        with Database() as db:
            try:
                db.execute(
                    """UPDATE threads SET title = %s WHERE id = %s""",
                    (title, thread_id)
                )
                return True
            except Exception as e:
                logger.error(f"Failed to update thread title: {str(e)}", exc_info=True)
                return False

class ConversationManager:
    """Conversation 관리를 담당하는 클래스"""
    
    @staticmethod
    def create_conversation(thread_id: int) -> int:
        """새로운 conversation 생성"""
        with Database() as db:
            db.connection.start_transaction()
            try:
                # 다음 conversation_id 가져오기
                db.execute(
                    """SELECT COALESCE(MAX(conversation_id), 0) + 1 as next_id 
                       FROM conversations WHERE thread_id = %s""",
                    (thread_id,)
                )
                next_id = db.fetchone()['next_id']
                
                # conversation 생성
                db.execute(
                    """INSERT INTO conversations (thread_id, conversation_id) 
                       VALUES (%s, %s)""",
                    (thread_id, next_id)
                )
                db.connection.commit()
                logger.info(f"Created conversation: {next_id}", 
                          extra={"thread_id": thread_id, "conversation_id": next_id})
                return next_id
            except Exception as e:
                db.connection.rollback()
                logger.error(f"Failed to create conversation: {str(e)}", exc_info=True)
                raise

    @staticmethod
    def save_conversation_data(db, thread_id: int, conversation_id: int, 
                             category: str, data: str) -> None:
        """conversation 데이터 저장"""
        try:
            db.execute(
                """INSERT INTO conversation_data (thread_id, conversation_id, category, data) 
                   VALUES (%s, %s, %s, %s)""",
                (thread_id, conversation_id, category, data)
            )
            logger.debug(f"Saved conversation data: {category}", 
                        extra={"thread_id": thread_id, "conversation_id": conversation_id})
        except Exception as e:
            logger.error(f"Failed to save conversation data: {str(e)}", exc_info=True)
            raise

class OpenAIAssistantManager:
    """OpenAI Assistant 관리를 담당하는 클래스"""
    
    gpt4o_assistant = None
    current_openai_thread_id = None
    TIMEOUT = 60  # API 타임아웃 설정

    @classmethod
    def initialize_assistant(cls) -> bool:
        """Assistant 초기화"""
        try:
            if hasattr(Config, 'GPT4O_ASSISTANT_ID') and Config.GPT4O_ASSISTANT_ID:
                try:
                    cls.gpt4o_assistant = openai_client.beta.assistants.retrieve(
                        Config.GPT4O_ASSISTANT_ID
                    )
                    logger.info(f"Retrieved GPT-4o Assistant: {cls.gpt4o_assistant.id}")
                except Exception as e:
                    logger.error(f"Failed to retrieve Assistant: {str(e)}", exc_info=True)
                    cls.gpt4o_assistant = None
                    
            if cls.gpt4o_assistant is None:
                cls.gpt4o_assistant = openai_client.beta.assistants.create(
                    name="Story Expander",
                    instructions=Config.HYBRID_SYSTEM_PROMPT,
                    model=Config.GPT_4O_MODEL
                )
                logger.info(f"Created new GPT-4o Assistant: {cls.gpt4o_assistant.id}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize assistant: {str(e)}", exc_info=True)
            return False

    @classmethod
    def create_or_get_thread(cls, existing_thread_id: Optional[str] = None) -> str:
        """OpenAI thread 생성 또는 가져오기"""
        if existing_thread_id:
            cls.current_openai_thread_id = existing_thread_id
            return existing_thread_id
        
        thread = openai_client.beta.threads.create()
        cls.current_openai_thread_id = thread.id
        return thread.id

class StoryGenerator:
    """이야기 생성을 담당하는 클래스"""

    @classmethod
    def generate_story_and_title(cls, user_input: str, 
                               existing_thread_id: Optional[str] = None) -> Generator:
        """이야기와 제목 생성"""
        try:
            expanded_story = ""
            openai_thread_id = OpenAIAssistantManager.create_or_get_thread(existing_thread_id)
            
            for content in cls._expand_story(user_input, openai_thread_id):
                expanded_story += content
                yield {
                    "msg": "expanding",
                    "content": content
                }
            
            if not expanded_story:
                raise Exception("이야기 생성 실패")
            
            result = cls._generate_title_and_recommendations(expanded_story, user_input)
            
            yield {
                "msg": "completed",
                "content": {
                    "story": expanded_story,
                    "title": result["title"],
                    "recommendations": result["recommendations"]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate story: {str(e)}", exc_info=True)
            raise

    @classmethod
    def _expand_story(cls, base_story: str, thread_id: str) -> Generator:
        """GPT-4를 사용한 이야기 확장"""
        try:
            openai_client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=base_story
            )
            
            run = openai_client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=OpenAIAssistantManager.gpt4o_assistant.id
            )
            
            start_time = time.time()
            
            while time.time() - start_time < OpenAIAssistantManager.TIMEOUT:
                run_status = openai_client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if run_status.status == 'completed':
                    messages = openai_client.beta.threads.messages.list(thread_id=thread_id)
                    for message in messages.data:
                        if message.role == "assistant":
                            yield message.content[0].text.value
                            return
                            
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    raise Exception(f"Story expansion failed: {run_status.status}")
                
                time.sleep(1)
            
            raise Exception("Story expansion timed out")
                
        except Exception as e:
            logger.error(f"Failed to expand story: {str(e)}", exc_info=True)
            raise

    @classmethod
    def _generate_title_and_recommendations(cls, content: str, theme: str) -> Dict[str, Any]:
        """제목과 추천 이야기 생성"""
        try:
            completion = openai_client.chat.completions.create(
                model=Config.GPT_MINI_MODEL,
                messages=[
                    {"role": "system", "content": """다음 형식의 JSON으로 응답해주세요:
{
    "title": "13글자 이하의 제목",
    "recommendations": ["추천1 이야기", "추천2 이야기", "추천3 이야기"]
}

요구사항:
1. 제목 생성: 주어진 고전소설 단락에 어울리는 간략한 제목을 생성하세요. 제목은 13글자를 넘기지 마세요.
2. 추천 생성: 주어진 주제와 비슷한 새로운 고전소설 줄거리 생성을 위한 이야기 소스를 3개 생성하세요.
   각 추천은 한 문장으로, "~~이야기"로 끝나야 합니다.
   영어를 사용하지 마세요."""},
                    {"role": "user", "content": f"단락: {content}\n\n주제: {theme}"}
                ],
                response_format={"type": "json_object"}
            )
            
            response_data = json.loads(completion.choices[0].message.content)
            return {
                "title": response_data.get("title", "제목 없음"),
                "recommendations": response_data.get("recommendations", ["", "", ""])[:3]
            }
        except Exception as e:
            logger.error(f"Failed to generate title and recommendations: {str(e)}", 
                        exc_info=True)
            return {
                "title": "제목 생성 오류",
                "recommendations": ["", "", ""]
            }

class StoryService:
    """전체 스토리 서비스를 관리하는 메인 클래스"""
    
    @classmethod
    def initialize(cls) -> bool:
        """서비스 초기화"""
        return OpenAIAssistantManager.initialize_assistant()

    @classmethod
    def hybrid_generate_story_with_assistant(cls, data: Dict[str, Any]) -> Generator:
        """하이브리드 방식으로 이야기 생성 및 스트리밍"""
        request_id = generate_session_hash()  # 요청 추적용 ID
        logger.info(f"Starting story generation", extra={"request_id": request_id})
        
        try:
            user_id = data.get('user_id')
            if not user_id:
                raise Exception("유효한 사용자 ID가 없습니다.")
            
            existing_db_thread_id = data.get('thread_id')
            thread_id = existing_db_thread_id
            
            # 새 thread 생성
            if thread_id is None:
                openai_thread_id = OpenAIAssistantManager.create_or_get_thread()
                thread_id = ThreadManager.create_thread(user_id, "새 이야기", openai_thread_id)
            
            # 이야기 생성
            for result in StoryGenerator.generate_story_and_title(
                data.get('user_input', ''),
                OpenAIAssistantManager.current_openai_thread_id
            ):
                if result["msg"] == "completed":
                    story_data = result["content"]
                    
                    # 데이터베이스 저장
                    with Database() as db:
                        db.connection.start_transaction()
                        try:
                            # 새 thread인 경우에만 제목 업데이트
                            if existing_db_thread_id is None:
                                ThreadManager.update_thread_title(thread_id, story_data["title"])
                            
                            conversation_id = ConversationManager.create_conversation(thread_id)
                            
                            # 대화 데이터 저장
                            data_items = [
                                ('user_input', data.get('user_input', '')),
                                ('tags', json.dumps(data.get('tags', {}), ensure_ascii=False)),
                                ('created_content', story_data["story"]),
                                ('created_title', story_data["title"])
                            ]
                            
                            for category, value in data_items:
                                ConversationManager.save_conversation_data(
                                    db, thread_id, conversation_id, category, value
                                )
                            
                            # 추천 데이터 저장
                            for i, rec in enumerate(story_data["recommendations"][:3], 1):
                                ConversationManager.save_conversation_data(
                                    db, thread_id, conversation_id, 
                                    f'recommended_{i}', rec
                                )
                            
                            db.connection.commit()
                            logger.info("Successfully saved all data", 
                                      extra={"request_id": request_id})
                            
                            # 응답 생성
                            response = {
                                "created_content": story_data["story"],
                                "created_title": story_data["title"],
                                "recommendations": story_data["recommendations"],
                                "thread_id": thread_id,
                                "conversation_id": conversation_id,
                                "user_id": user_id
                            }
                            
                            if existing_db_thread_id is None:
                                response["openai_thread_id"] = OpenAIAssistantManager.current_openai_thread_id
                            
                            yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"
                            
                        except Exception as e:
                            db.connection.rollback()
                            logger.error(f"Database transaction failed: {str(e)}", 
                                       extra={"request_id": request_id}, exc_info=True)
                            raise e
                else:
                    yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                    
        except Exception as e:
            logger.error(f"Story generation failed: {str(e)}", 
                        extra={"request_id": request_id}, exc_info=True)
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"