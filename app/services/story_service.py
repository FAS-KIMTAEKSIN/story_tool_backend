from openai import OpenAI
from app.config import Config
from app.utils.database import Database
from app.utils.helpers import generate_session_hash
import json
import traceback
import time
import requests
import logging
from typing import Optional, Dict, Any, Generator, List

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 스트림 핸들러 추가
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

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

    @staticmethod
    def _extract_message_content(message) -> Optional[str]:
        """메시지에서 안전하게 텍스트 내용을 추출"""
        try:
            logger.debug(f"Message object structure: {message}")
            logger.debug(f"Message content type: {type(message.content)}")
            
            if not message.content or len(message.content) == 0:
                logger.warning(f"Message content is empty or None: {message.content}")
                return None
            
            logger.debug(f"Message content length: {len(message.content)}")
            for i, content in enumerate(message.content):
                logger.debug(f"Content {i} type: {type(content)}")
                logger.debug(f"Content {i} structure: {content}")
                if hasattr(content, 'text') and hasattr(content.text, 'value'):
                    return content.text.value
            
            logger.warning("No text content found in message")
            return None
        except Exception as e:
            logger.warning(f"Failed to extract message content: {str(e)}", exc_info=True)
            return None

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
            logger.info("=== Story Expansion Started ===")
            logger.info(f"Thread ID: {thread_id}")
            logger.info(f"Base story length: {len(base_story)}")
            
            # 사용자 메시지 전송
            logger.info("Creating user message...")
            try:
                user_message = openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=base_story
                )
                logger.info(f"User message created successfully: {user_message.id}")
            except Exception as e:
                logger.error("Failed to create user message", exc_info=True)
                raise
            
            # 실행 시작 (스트리밍 모드)
            logger.info("Creating assistant run with streaming...")
            try:
                stream = openai_client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=OpenAIAssistantManager.gpt4o_assistant.id,
                    stream=True
                )
                logger.info("Stream created successfully")
            except Exception as e:
                logger.error("Failed to create stream", exc_info=True)
                raise
            
            current_content = ""
            message_id = None
            
            # 스트리밍 이벤트 처리
            for event in stream:
                event_type = getattr(event, 'event', None)
                logger.debug(f"Received event: {event_type}")
                
                if event_type == 'thread.message.delta':
                    delta = event.data.delta
                    if delta.content and len(delta.content) > 0:
                        content_item = delta.content[0]
                        if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                            text_value = content_item.text.value
                            current_content += text_value
                            logger.info(f"Received content chunk (length: {len(text_value)})")
                            yield text_value
                
                elif event_type == 'thread.message.completed':
                    if not message_id:
                        message_id = event.data.id
                        logger.info(f"Message completed: {message_id}")
                
                elif event_type == 'thread.run.completed':
                    logger.info("Run completed successfully")
                    if not current_content:
                        logger.error("No content generated during story expansion")
                        raise Exception("이야기 생성에 실패했습니다: 내용이 비어있습니다")
                    return
                
                elif event_type == 'thread.run.failed':
                    error_msg = getattr(event.data, 'last_error', 'Unknown error')
                    logger.error(f"Run failed: {error_msg}")
                    raise Exception(f"Story expansion failed: {error_msg}")
            
            if not current_content:
                logger.error("No content generated during story expansion")
                raise Exception("이야기 생성에 실패했습니다: 내용이 비어있습니다")
            
            logger.info("=== Story Expansion Completed Successfully ===")
            
        except Exception as e:
            logger.error(f"Story expansion failed: {str(e)}", exc_info=True)
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
1. 제목 생성: 다음 고전소설 단락에 어울리는 간략한 제목을 생성하세요. 제목은 13글자를 넘기지 마세요.
2. 추천 생성: 주어진 주제와 비슷한 새로운 고전소설 줄거리 생성을 위한 이야기 소스를 3개 생성하세요.
                     각 소스는 제목 없이 줄거리 생성을 위한 이야기 소스를 한 문장으로 생성할 것
   "~~이야기"로 끝낼 것. 예시처럼 단락의 마무리가 이야기로 끝나야해 (예시. 모험이 시작되는 이야기)
   영어를 절대 사용하지 말 것."""},
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
    
    _search_cache = {}  # 검색 결과 캐시
    
    @classmethod
    def initialize(cls) -> bool:
        """서비스 초기화"""
        return OpenAIAssistantManager.initialize_assistant()

    @classmethod
    def search_stories(cls, query: str, page: int = 1, per_page: int = 10) -> Dict[str, Any]:
        """이야기 검색"""
        try:
            with Database() as db:
                # 전체 결과 수 조회
                db.execute(
                    """SELECT COUNT(*) as total FROM threads t
                       JOIN conversation_data cd ON t.id = cd.thread_id
                       WHERE cd.category IN ('created_title', 'created_content')
                       AND (cd.data LIKE %s OR t.title LIKE %s)""",
                    (f"%{query}%", f"%{query}%")
                )
                total = db.fetchone()['total']

                # 페이지네이션된 결과 조회
                offset = (page - 1) * per_page
                db.execute(
                    """SELECT DISTINCT t.id, t.title, t.user_id,
                       MAX(CASE WHEN cd.category = 'created_content' THEN cd.data END) as content,
                       MAX(cd.conversation_id) as latest_conversation_id
                       FROM threads t
                       JOIN conversation_data cd ON t.id = cd.thread_id
                       WHERE cd.category IN ('created_title', 'created_content')
                       AND (cd.data LIKE %s OR t.title LIKE %s)
                       GROUP BY t.id, t.title, t.user_id
                       ORDER BY t.id DESC
                       LIMIT %s OFFSET %s""",
                    (f"%{query}%", f"%{query}%", per_page, offset)
                )
                results = db.fetchall()

                search_results = {
                    'total': total,
                    'page': page,
                    'per_page': per_page,
                    'total_pages': (total + per_page - 1) // per_page,
                    'results': results
                }

                # 검색 결과 캐시 업데이트
                cls.update_search_results(query, search_results)
                return search_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise

    @classmethod
    def search_and_recommend(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """검색 및 추천 결과 조회"""
        try:
            # 필수 파라미터 검증
            thread_id = data.get('thread_id')
            conversation_id = data.get('conversation_id')
            user_id = data.get('user_id')
            user_input = data.get('user_input', '')
            tags = data.get('tags', {})

            if not all([thread_id, conversation_id, user_id]):
                raise ValueError("Required parameters missing")

            with Database() as db:
                # 소유권 확인
                db.execute(
                    """SELECT 1 FROM threads t 
                       JOIN conversations c ON t.id = c.thread_id 
                       WHERE t.id = %s AND c.conversation_id = %s AND t.user_id = %s""",
                    (thread_id, conversation_id, user_id)
                )
                if not db.fetchone():
                    raise ValueError("Invalid thread_id, conversation_id, or unauthorized access")

                # 유사한 문서 검색
                search_results = []
                db.execute(
                    """SELECT DISTINCT t.id, t.title, cd.data as content
                       FROM threads t
                       JOIN conversation_data cd ON t.id = cd.thread_id
                       WHERE cd.category = 'created_content'
                       AND (t.title LIKE %s OR cd.data LIKE %s)
                       AND t.id != %s
                       LIMIT 3""",
                    (f"%{user_input}%", f"%{user_input}%", thread_id)
                )
                results = db.fetchall()
                
                # 검색 결과 처리
                for idx, result in enumerate(results or [], 1):
                    try:
                        search_results.append({
                            'document_id': idx,
                            'metadata': {
                                'id': result.get('id', 0),
                                'title': result.get('title', '제목 없음')
                            },
                            'content': result.get('content', ''),
                            'score': 1.0 - (idx * 0.1)  # 간단한 점수 계산
                        })
                    except Exception as e:
                        logger.error(f"Error processing search result: {str(e)}", exc_info=True)
                        continue

                # 임시 추천 문서 생성
                recommendations = [
                    f"추천 {i+1}: {user_input}와(과) 비슷한 이야기"
                    for i in range(3)
                ]

                # 결과를 클라이언트가 기대하는 형식으로 구성
                formatted_result = {
                    "success": True,
                    "result": {
                        "similar_1": search_results[0] if len(search_results) > 0 else {},
                        "similar_2": search_results[1] if len(search_results) > 1 else {},
                        "similar_3": search_results[2] if len(search_results) > 2 else {},
                        "recommended_1": recommendations[0] if len(recommendations) > 0 else "",
                        "recommended_2": recommendations[1] if len(recommendations) > 1 else "",
                        "recommended_3": recommendations[2] if len(recommendations) > 2 else ""
                    }
                }

                # 검색 결과 업데이트
                try:
                    cls.update_search_results(
                        thread_id=thread_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        search_results=search_results,
                        recommendations=recommendations
                    )
                except Exception as e:
                    logger.error(f"Failed to update search results: {str(e)}", exc_info=True)
                    # 업데이트 실패해도 검색 결과는 반환

                return formatted_result

        except Exception as e:
            logger.error(f"Search and recommend failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "result": {
                    "similar_1": {},
                    "similar_2": {},
                    "similar_3": {},
                    "recommended_1": "",
                    "recommended_2": "",
                    "recommended_3": ""
                }
            }

    @classmethod
    def update_search_results(cls, thread_id: str, conversation_id: str, user_id: int,
                            search_results: List[Dict[str, Any]], recommendations: List[str]) -> Dict[str, bool]:
        """검색 결과 업데이트"""
        try:
            with Database() as db:
                db.connection.start_transaction()
                try:
                    # 소유권 확인
                    db.execute(
                        """SELECT 1 FROM threads t 
                           JOIN conversations c ON t.id = c.thread_id 
                           WHERE t.id = %s AND c.conversation_id = %s AND t.user_id = %s""",
                        (thread_id, conversation_id, user_id)
                    )
                    if not db.fetchone():
                        raise Exception("Invalid thread_id, conversation_id, or unauthorized access")

                    # 검색 결과 업데이트
                    for idx, category in enumerate(['similar_1', 'similar_2', 'similar_3'], 1):
                        result = search_results[idx-1] if idx <= len(search_results) else {}
                        db.execute(
                            """INSERT INTO conversation_data 
                               (thread_id, conversation_id, category, data)
                               VALUES (%s, %s, %s, %s)
                               ON DUPLICATE KEY UPDATE data = VALUES(data)""",
                            (thread_id, conversation_id, category, 
                             json.dumps(result, ensure_ascii=False))
                        )

                    # 추천 문서 업데이트
                    for idx, category in enumerate(['recommended_1', 'recommended_2', 'recommended_3'], 1):
                        recommendation = recommendations[idx-1] if idx <= len(recommendations) else ""
                        db.execute(
                            """INSERT INTO conversation_data 
                               (thread_id, conversation_id, category, data)
                               VALUES (%s, %s, %s, %s)
                               ON DUPLICATE KEY UPDATE data = VALUES(data)""",
                            (thread_id, conversation_id, category, recommendation)
                        )

                    db.connection.commit()
                    logger.info(f"Successfully updated search results for thread_id: {thread_id}, conversation_id: {conversation_id}, user_id: {user_id}")
                    return {"success": True}

                except Exception as e:
                    db.connection.rollback()
                    logger.error(f"Database transaction failed: {str(e)}", exc_info=True)
                    raise

        except Exception as e:
            logger.error(f"Failed to update search results: {str(e)}", exc_info=True)
            return {"success": False}

    @classmethod
    def get_search_results(cls, query: str, thread_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """캐시된 검색 결과 조회"""
        cache_key = f"{query}_{thread_id}" if thread_id else query
        cache_data = cls._search_cache.get(cache_key)
        if cache_data and time.time() - cache_data['timestamp'] < 300:  # 5분 캐시
            return cache_data['results']
        return None

    @classmethod
    def clear_search_cache(cls) -> None:
        """검색 결과 캐시 초기화"""
        cls._search_cache.clear()

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
            existing_openai_thread_id = None
            
            # 기존 thread의 openai_thread_id 조회
            if thread_id is not None:
                with Database() as db:
                    db.execute(
                        "SELECT thread_id FROM threads WHERE id = %s",
                        (thread_id,)
                    )
                    result = db.fetchone()
                    if result:
                        existing_openai_thread_id = result['thread_id']
                        logger.info(f"Found existing OpenAI thread ID: {existing_openai_thread_id}")
            
            # 새 thread 생성 또는 기존 thread 사용
            if thread_id is None:
                openai_thread_id = OpenAIAssistantManager.create_or_get_thread()
                thread_id = ThreadManager.create_thread(user_id, "새 이야기", openai_thread_id)
            else:
                openai_thread_id = OpenAIAssistantManager.create_or_get_thread(existing_openai_thread_id)
            
            # 이야기 생성
            for result in StoryGenerator.generate_story_and_title(
                data.get('user_input', ''),
                openai_thread_id
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
                                "user_id": user_id,
                                "openai_thread_id": openai_thread_id
                            }
                            
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