from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from app.config import Config
from app.utils.database import Database
import json
from mysql.connector import Error
import requests
from app.utils.helpers import generate_session_hash
import traceback
import time

class StoryService:
    client = OpenAI()
    
    # Assistant 객체 초기화
    title_assistant = None
    recommendation_assistant = None
    fine_tuned_assistant = None
    gpt4o_assistant = None
    
    # 현재 사용 중인 OpenAI 쓰레드 ID를 저장할 클래스 변수
    current_openai_thread_id = None
    
    @classmethod
    def initialize_assistants(cls):
        """Assistant 객체 초기화 메소드"""
        try:
            # GPT-4o 확장용 Assistant만 초기화
            if hasattr(Config, 'GPT4O_ASSISTANT_ID') and Config.GPT4O_ASSISTANT_ID:
                try:
                    cls.gpt4o_assistant = cls.client.beta.assistants.retrieve(Config.GPT4O_ASSISTANT_ID)
                    print(f"GPT-4o Assistant retrieved: {cls.gpt4o_assistant.id}")
                except Exception as e:
                    print(f"Error retrieving GPT-4o Assistant: {str(e)}")
                    cls.gpt4o_assistant = None
                    
            if cls.gpt4o_assistant is None:
                cls.gpt4o_assistant = cls.client.beta.assistants.create(
                    name="Story Expander",
                    instructions=Config.HYBRID_SYSTEM_PROMPT,
                    model=Config.GPT_4O_MODEL
                )
                print(f"New GPT-4o Assistant created: {cls.gpt4o_assistant.id}")
                print(f"GPT4O_ASSISTANT_ID={cls.gpt4o_assistant.id}")
            
            print(f"Assistants initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing assistants: {str(e)}")
            return False

    @classmethod
    def generate_title(cls, content):
        """제목 생성"""
        completion = cls.client.chat.completions.create(
            model=Config.GPT_MINI_MODEL,
            messages=[
                {"role": "system", "content": "다음 고전소설 단락에 어울리는 **간략한 제목**을 생성해. **제목은 절대로 13글자를 넘기지 마.**"},
                {"role": "user", "content": content}
            ]
        )
        title = completion.choices[0].message.content
        # 양쪽에 따옴표가 있는 경우에만 제거 (큰따옴표 또는 작은따옴표)
        if (title.startswith('"') and title.endswith('"')) or \
           (title.startswith("'") and title.endswith("'")):
            return title[1:-1]
        return title

    @classmethod
    def generate_title_with_assistant(cls, content):
        """제목 생성 (Assistant API + Structured Output 사용)"""
        try:
            # Assistant가 초기화되지 않았다면 초기화
            if cls.title_assistant is None:
                cls.initialize_assistants()
                
            if cls.title_assistant is None:
                raise Exception("제목 생성 Assistant 초기화 실패")
            
            # Thread 생성
            thread = cls.client.beta.threads.create()
            
            # 메시지 추가
            cls.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=content
            )
            
            # Run 생성 및 완료 대기
            run = cls.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=cls.title_assistant.id
            )
            
            # Run 완료 대기
            while True:
                run_status = cls.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == 'completed':
                    break
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    raise Exception(f"Title generation failed with status: {run_status.status}")
                time.sleep(0.5)
            
            # 결과 가져오기
            run_steps = cls.client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id
            )
            
            for step in run_steps.data:
                if hasattr(step, 'step_details') and hasattr(step.step_details, 'tool_calls'):
                    for tool_call in step.step_details.tool_calls:
                        if tool_call.type == 'function' and tool_call.function.name == 'generate_title':
                            try:
                                output = json.loads(tool_call.function.output)
                                return output.get('title', "제목 없음")
                            except Exception as e:
                                print(f"Error parsing title output: {str(e)}")
            
            # 정형화된 출력이 없는 경우, 메시지에서 직접 추출
            messages = cls.client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            for message in messages.data:
                if message.role == "assistant":
                    return message.content[0].text.value
            
            return "제목 없음"  # 실패 시 기본값
            
        except Exception as e:
            print(f"Error generating title with assistant: {str(e)}")
            return "제목 생성 오류"

    @classmethod
    def generate_recommendation(cls, theme):
        """추천 이야기 생성"""
        completion = cls.client.chat.completions.create(
            model=Config.GPT_MINI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """다음 내용과 비슷한 새로운 고전소설 줄거리 생성을 위한 이야기 소스를 한 문장으로 생성해.
                                다음 지침을 따라.
                                1. 제목 없이 줄거리 생성을 위한 이야기 소스를 한 문장으로 생성할 것
                                2. ~~이야기 로 끝낼 것. 예시처럼 단락의 마무리가 이야기로 끝나야해.  (예시. 모험이 시작되는 이야기)
                                3. 영어를 절대 사용지 말 것"""
                },
                {"role": "user", "content": theme}
            ]
        )
        return completion.choices[0].message.content

    @classmethod
    def generate_recommendation_with_assistant(cls, theme):
        """추천 이야기 생성 (Assistant API + Structured Output 사용)"""
        try:
            # Assistant가 초기화되지 않았다면 초기화
            if cls.recommendation_assistant is None:
                cls.initialize_assistants()
                
            if cls.recommendation_assistant is None:
                raise Exception("추천 생성 Assistant 초기화 실패")
            
            # Thread 생성
            thread = cls.client.beta.threads.create()
            
            # 메시지 추가
            cls.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=theme
            )
            
            # Run 생성 및 완료 대기
            run = cls.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=cls.recommendation_assistant.id
            )
            
            # Run 완료 대기
            timeout = 60  # 60초 타임아웃
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                run_status = cls.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == 'completed':
                    break
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    raise Exception(f"Recommendation generation failed with status: {run_status.status}")
                time.sleep(0.5)
            
            if time.time() - start_time >= timeout:
                raise Exception("Recommendation generation timed out")
            
            # 결과 가져오기
            run_steps = cls.client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id
            )
            
            for step in run_steps.data:
                if hasattr(step, 'step_details') and hasattr(step.step_details, 'tool_calls'):
                    for tool_call in step.step_details.tool_calls:
                        if tool_call.type == 'function' and tool_call.function.name == 'generate_recommendation':
                            try:
                                output = json.loads(tool_call.function.output)
                                return output.get('recommendation', "추천 이야기를 생성할 수 없습니다.")
                            except Exception as e:
                                print(f"Error parsing recommendation output: {str(e)}")
            
            # 정형화된 출력이 없는 경우, 메시지에서 직접 추출
            messages = cls.client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            for message in messages.data:
                if message.role == "assistant":
                    return message.content[0].text.value
            
            return "추천 이야기를 생성할 수 없습니다."  # 실패 시 기본값
            
        except Exception as e:
            print(f"Error generating recommendation with assistant: {str(e)}")
            return "추천 이야기 생성 오류"

    @classmethod
    def _save_openai_thread_to_db(cls, user_id, openai_thread_id):
        """OpenAI 쓰레드 ID를 DB에 저장"""
        try:
            with Database() as cursor:
                # threads 테이블에 새 항목 추가
                cursor.execute(
                    """INSERT INTO threads 
                       (user_id, title, thread_id) 
                       VALUES (%s, %s, %s)""",
                    (user_id, "새 이야기", openai_thread_id)
                )
                
                db_thread_id = cursor.lastrowid
                print(f"[DEBUG] Saved OpenAI thread {openai_thread_id} as DB thread {db_thread_id}")
                return db_thread_id
                
        except Exception as e:
            print(f"[ERROR] Failed to save OpenAI thread to DB: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            return None

    @classmethod
    def _get_db_thread_id_from_openai_thread(cls, user_id, openai_thread_id):
        """OpenAI 쓰레드 ID로부터 DB ID 조회"""
        try:
            with Database() as cursor:
                cursor.execute(
                    """SELECT id FROM threads 
                       WHERE user_id = %s AND thread_id = %s
                       LIMIT 1""",
                    (user_id, openai_thread_id)
                )
                result = cursor.fetchone()
                if result:
                    return result['id']
                return None
        except Exception as e:
            print(f"[ERROR] Failed to get DB thread ID: {str(e)}")
            return None

    @classmethod
    def save_to_database(cls, user_id, data, result):
        """생성된 이야기를 데이터베이스에 저장"""
        try:
            with Database() as db:
                # 트랜잭션 시작 - Database 클래스에서 처리해야 함
                # 먼저 thread 생성
                openai_thread_id = getattr(cls, 'current_openai_thread_id', None)
                title = result.get('created_title', '새 이야기')
                
                db.execute(
                    """INSERT INTO threads (user_id, title, thread_id) VALUES (%s, %s, %s)""",
                    (user_id, title, openai_thread_id)
                )
                db_thread_id = db.lastrowid()
                
                # conversation 추가
                db.execute(
                    """INSERT INTO conversations (thread_id, conversation_id) 
                       VALUES (%s, %s)""",
                    (db_thread_id, 1)  # 첫 번째 대화는 1
                )
                
                # conversation_data에 컨텐츠 데이터 추가
                cls._insert_conversation_data_batch(db, db_thread_id, 1, data, result)
                
                print(f"[INFO] Saved to database: thread_id={db_thread_id}, conversation_id=1")
                
                # 변수 초기화
                cls.current_openai_thread_id = None
                
                return db_thread_id, 1
                
        except Exception as e:
            print(f"[ERROR] Failed to save to database: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            return None, None

    @classmethod
    def _insert_conversation_data_batch(cls, db, thread_id, conversation_id, data, result):
        """대화 데이터를 일괄 저장"""
        try:
            # 사용자 입력
            db.execute(
                """INSERT INTO conversation_data (thread_id, conversation_id, category, data) 
                   VALUES (%s, %s, %s, %s)""",
                (thread_id, conversation_id, 'user_input', data.get('user_input', ''))
            )
            
            # 태그 정보
            db.execute(
                """INSERT INTO conversation_data (thread_id, conversation_id, category, data) 
                   VALUES (%s, %s, %s, %s)""",
                (thread_id, conversation_id, 'tags', json.dumps(data.get('tags', {}), ensure_ascii=False))
            )
            
            # 제목
            db.execute(
                """INSERT INTO conversation_data (thread_id, conversation_id, category, data) 
                   VALUES (%s, %s, %s, %s)""",
                (thread_id, conversation_id, 'created_title', result.get('created_title', ''))
            )
            
            # 생성된 내용
            db.execute(
                """INSERT INTO conversation_data (thread_id, conversation_id, category, data) 
                   VALUES (%s, %s, %s, %s)""",
                (thread_id, conversation_id, 'created_content', result.get('created_content', ''))
            )
            
            # 추천 저장
            recommendations = result.get('recommendations', [])
            for i, rec in enumerate(recommendations[:3], 1):
                db.execute(
                    """INSERT INTO conversation_data (thread_id, conversation_id, category, data) 
                       VALUES (%s, %s, %s, %s)""",
                    (thread_id, conversation_id, f'recommended_{i}', rec)
                )
        except Exception as e:
            print(f"[ERROR] Failed to insert conversation data: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            raise

    @classmethod
    def add_conversation_to_thread(cls, thread_id, user_id, data, result):
        """기존 쓰레드에 새 대화 추가"""
        connection = None
        try:
            with Database() as cursor:
                connection = cursor.connection
                
                # transactions 시작
                connection.start_transaction()
                
                # 다음 conversation_id 가져오기
                next_id = Database.get_next_conversation_id(cursor, thread_id)
                
                # conversation 추가
                cursor.execute(
                    """INSERT INTO conversations (thread_id, conversation_id) 
                       VALUES (%s, %s)""",
                    (thread_id, next_id)
                )
                
                # conversation_data에 컨텐츠 데이터 추가
                cls._insert_conversation_data(cursor, thread_id, next_id, 'user_input', data.get('user_input', ''))
                cls._insert_conversation_data(cursor, thread_id, next_id, 'tags', json.dumps(data.get('tags', {}), ensure_ascii=False))
                cls._insert_conversation_data(cursor, thread_id, next_id, 'created_title', result.get('created_title', ''))
                cls._insert_conversation_data(cursor, thread_id, next_id, 'created_content', result.get('created_content', ''))
                
                # 추천 저장
                recommendations = result.get('recommendations', [])
                for i, rec in enumerate(recommendations[:3], 1):
                    cls._insert_conversation_data(cursor, thread_id, next_id, f'recommended_{i}', rec)
                
                # 스레드 제목 업데이트 (첫 번째 대화가 아닌 경우)
                if next_id > 1:
                    cursor.execute(
                        """UPDATE threads SET title = %s WHERE thread_id = %s""",
                        (result.get('created_title', '새 이야기'), thread_id)
                    )
                
                connection.commit()
                print(f"[INFO] Added to thread: thread_id={thread_id}, conversation_id={next_id}")
                
                return next_id
                
        except Exception as e:
            if connection:
                connection.rollback()
            print(f"[ERROR] Failed to add conversation: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            return None

    @classmethod
    def generate_story(cls, data):
        """이야기 생성 및 스트리밍"""
        try:
            # 이야기 생성
            content = None
            generated_result = None
            
            for generated_content in cls._stream_story_generation(data):
                try:
                    # JSON 문자열인 경우 파싱
                    if isinstance(generated_content, str):
                        parsed_data = json.loads(generated_content)
                        # 중간 생성 과정 전달
                        yield generated_content
                        
                        # 최종 결과인 경우 저장
                        if parsed_data.get('msg') == 'process_completed':
                            if parsed_data.get('output') and parsed_data['output'].get('data'):
                                content = parsed_data['output']['data'][0]
                    # 최종 컨텐츠인 경우
                    else:
                        content = generated_content
                except json.JSONDecodeError:
                    continue
            
            if not content:
                raise Exception("이야기 생성 실패: 컨텐츠가 비어있음")
            
            # 제목 생성
            title = cls.generate_title_with_assistant(content)
            
            # 최종 결과를 딕셔너리로 yield
            generated_result = {
                "created_title": title,
                "created_content": content
            }
            yield generated_result
            
        except Exception as e:
            print(f"[DEBUG] ERROR in generate_story: {str(e)}")
            print(f"[DEBUG] ERROR traceback: {traceback.format_exc()}")
            raise e

    @classmethod
    def _stream_story_generation(cls, data):
        """VAIV API를 사용한 이야기 생성 스트리밍"""
        from app.services.search_service import SearchService
        
        # 프롬프트 생성
        prompt = SearchService.process_input(data)
        print(f"[DEBUG] Starting generation with prompt: {prompt}")
        
        # VAIV API 호출 및 스트리밍 처리
        session_hash = generate_session_hash()
        print(f"[DEBUG] Generated session hash: {session_hash}")
        
        # 첫 번째 API 요청 - 큐 참여
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "data": [prompt],
            "event_data": None,
            "fn_index": 0,
            "trigger_id": 4,
            "session_hash": session_hash
        }
        
        print(f"[DEBUG] Sending join request with payload: {payload}")
        response = requests.post(Config.VAIV_QUEUE_JOIN_URL, headers=headers, json=payload)
        print(f"[DEBUG] Join response status: {response.status_code}")
        print(f"[DEBUG] Join response content: {response.text}")
        
        if response.status_code != 200:
            raise Exception("API 요청 실패")
        
        # 두 번째 API 요청 - 데이터 가져오기
        data_url = f'{Config.VAIV_QUEUE_DATA_URL}?session_hash={session_hash}'
        headers = {
            'Accept': 'text/event-stream',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive'
        }
        
        print(f"[DEBUG] Starting stream request to: {data_url}")
        with requests.get(data_url, headers=headers, stream=True) as response:
            print(f"[DEBUG] Stream response status: {response.status_code}")
            
            if response.status_code != 200:
                raise Exception("데이터 가져오기 실패")
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(f"[DEBUG] Received line: {decoded_line}")
                    
                    if decoded_line.startswith('data: '):
                        try:
                            data = json.loads(decoded_line[6:])
                            print(f"[DEBUG] Parsed data: {data}")
                            
                            if isinstance(data, dict):
                                # 그라디오 응답을 그대로 전달
                                yield decoded_line[6:]  # 'data: ' 제외하고 전달
                                
                                # process_completed 메시지에서 최종 컨텐츠 저장
                                if data.get('msg') == 'process_completed':
                                    if data.get('output') and data['output'].get('data'):
                                        content = data['output']['data'][0]
                                        if content:  # 컨텐츠가 있는 경우에만 yield
                                            yield content
                                    break
                                    
                        except json.JSONDecodeError as e:
                            print(f"[DEBUG] JSON decode error: {e}")
                            continue

    @classmethod
    def update_search_results(cls, thread_id, conversation_id, user_id, search_results=None, recommendations=None):
        """검색 결과를 데이터베이스에 업데이트"""
        try:
            # DB 트랜잭션 시작
            with Database() as cursor:
                # 검색 결과 업데이트
                if search_results:
                    for idx, result in enumerate(search_results, 1):
                        if idx > 3:  # 최대 3개만 처리
                            break
                        
                        category = f"similar_{idx}"
                        
                        # 기존 데이터 확인
                        cursor.execute(
                            """SELECT id FROM conversation_data 
                               WHERE thread_id = %s AND conversation_id = %s AND category = %s""",
                            (thread_id, conversation_id, category)
                        )
                        existing = cursor.fetchone()
                        
                        # JSON 형식으로 변환
                        data_json = json.dumps(result, ensure_ascii=False)
                        
                        if existing:
                            # 업데이트
                            cursor.execute(
                                """UPDATE conversation_data 
                                   SET data = %s 
                                   WHERE id = %s""",
                                (data_json, existing['id'])
                            )
                        else:
                            # 삽입
                            cursor.execute(
                                """INSERT INTO conversation_data 
                                   (thread_id, conversation_id, category, data) 
                                   VALUES (%s, %s, %s, %s)""",
                                (thread_id, conversation_id, category, data_json)
                            )
                
                # 추천 결과 업데이트
                if recommendations:
                    for idx, recommendation in enumerate(recommendations, 1):
                        if idx > 3:  # 최대 3개만 처리
                            break
                        
                        category = f"recommended_{idx}"
                        
                        # 기존 데이터 확인
                        cursor.execute(
                            """SELECT id FROM conversation_data 
                               WHERE thread_id = %s AND conversation_id = %s AND category = %s""",
                            (thread_id, conversation_id, category)
                        )
                        existing = cursor.fetchone()
                        
                        if existing:
                            # 업데이트
                            cursor.execute(
                                """UPDATE conversation_data 
                                   SET data = %s 
                                   WHERE id = %s""",
                                (recommendation, existing['id'])
                            )
                        else:
                            # 삽입
                            cursor.execute(
                                """INSERT INTO conversation_data 
                                   (thread_id, conversation_id, category, data) 
                                   VALUES (%s, %s, %s, %s)""",
                                (thread_id, conversation_id, category, recommendation)
                            )
            
            return {"success": True}
        except Exception as e:
            print(f"[ERROR] Failed to update search results: {str(e)}")
            return {"success": False, "error": str(e)}

    @classmethod
    def _expand_with_gpt4o(cls, base_story, user_id=None, existing_thread_id=None):
        """GPT-4o로 스토리 확장 (Assistant API 사용)"""
        try:
            # Assistant가 초기화되지 않았다면 초기화
            if cls.gpt4o_assistant is None:
                cls.initialize_assistants()
                
            if cls.gpt4o_assistant is None:
                raise Exception("GPT-4o Assistant 초기화 실패")
            
            # 쓰레드 생성 또는 기존 쓰레드 사용
            if existing_thread_id:
                # 기존 쓰레드 사용
                thread_id = existing_thread_id
                cls.current_openai_thread_id = existing_thread_id  # 현재 쓰레드 ID 저장
                print(f"[DEBUG] Using existing thread: {thread_id}")
            else:
                # 새 쓰레드 생성
                thread = cls.client.beta.threads.create()
                thread_id = thread.id
                cls.current_openai_thread_id = thread_id  # 현재 쓰레드 ID 저장
                print(f"[DEBUG] Created new thread: {thread_id}")
                
                # 새로 생성된 쓰레드 ID를 즉시 DB에 저장 (user_id가 제공된 경우)
                if user_id:
                    cls._save_openai_thread_to_db(user_id, thread_id)
            
            # 메시지 추가
            cls.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=base_story
            )
            
            # Run 생성
            run = cls.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=cls.gpt4o_assistant.id
            )
            
            # Run 상태 확인 및 대기
            expanded_story = ""
            buffer = ""
            
            print(f"[DEBUG] Waiting for GPT-4o expansion to complete...")
            
            while True:
                run_status = cls.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )
                
                if run_status.status == 'completed':
                    # 완료된 경우 메시지 가져오기
                    messages = cls.client.beta.threads.messages.list(
                        thread_id=thread_id
                    )
                    
                    for message in messages.data:
                        if message.role == "assistant":
                            expanded_story = message.content[0].text.value
                            yield expanded_story
                            return
                            
                elif run_status.status in ['failed', 'cancelled', 'expired']:
                    print(f"[ERROR] GPT-4o expansion failed with status: {run_status.status}")
                    if hasattr(run_status, 'last_error'):
                        print(f"[ERROR] Last error: {run_status.last_error}")
                    return
                
                # 아직 진행 중인 경우 잠시 대기
                time.sleep(1)
                buffer += "."
                if len(buffer) >= 5:
                    yield buffer
                    buffer = ""
            
        except Exception as e:
            print(f"[ERROR] Error in GPT-4o expansion: {str(e)}")
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return None

    @classmethod
    def _format_hybrid_prompt(cls, data):
        """하이브리드 생성용 프롬프트 포맷팅"""
        from app.services.search_service import SearchService
        
        theme = data.get('user_input', '')
        tags = data.get('tags', {})
        keywords = SearchService.extract_keywords_from_theme(theme)
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": Config.STORY_GENERATION_PROMPT
                },
                {
                    "role": "user",
                    "content": f"[내용 분류]\n{tags}\n\n[주제어]\n{keywords}\n\n[주제문]\n{theme}"
                }
            ]
        }

    @classmethod
    def _generate_with_fine_tuned_model(cls, prompt):
        """파인튜닝된 모델로 기본 스토리 생성 (Chat Completions API 사용)"""
        try:
            # Chat Completions API를 사용하여 스토리 생성
            response = cls.client.chat.completions.create(
                model=Config.FINE_TUNED_MODEL,
                messages=prompt["messages"],
                temperature=0.7,
                max_tokens=1024
            )
            
            # 응답에서 스토리 추출
            generated_story = response.choices[0].message.content.strip()
            
            if not generated_story:
                print("[ERROR] Fine-tuned model returned empty response")
                return None
            
            return generated_story
            
        except Exception as e:
            print(f"[ERROR] Error in fine-tuned model generation: {str(e)}")
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return None

    @classmethod
    def generate_title_and_recommendations(cls, content, theme):
        """제목과 추천 이야기를 한 번의 API 호출로 생성"""
        try:
            completion = cls.client.chat.completions.create(
                model=Config.GPT_MINI_MODEL,
                messages=[
                    {"role": "system", "content": """두 가지 작업을 수행해주세요:
                    1. 제목 생성: 주어진 고전소설 단락에 어울리는 간략한 제목을 생성하세요. 제목은 13글자를 넘기지 마세요.
                    2. 추천 생성: 주어진 주제와 비슷한 새로운 고전소설 줄거리 생성을 위한 이야기 소스를 3개 생성하세요.
                       각 추천은 한 문장으로, "~~이야기"로 끝나야 합니다. (예: "모험이 시작되는 이야기")
                       영어를 사용하지 마세요.
                    
                    JSON 형식으로 응답해주세요:
                    {
                        "title": "생성된 제목",
                        "recommendations": ["추천1", "추천2", "추천3"]
                    }"""},
                    {"role": "user", "content": f"단락: {content}\n\n주제: {theme}"}
                ],
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            response_data = json.loads(response_text)
            
            title = response_data.get("title", "제목 없음")
            recommendations = response_data.get("recommendations", [])
            
            # 추천이 3개 미만인 경우 빈 문자열로 채우기
            while len(recommendations) < 3:
                recommendations.append("")
            
            return {
                "title": title,
                "recommendations": recommendations[:3]  # 최대 3개만 사용
            }
        except Exception as e:
            print(f"[ERROR] Error generating title and recommendations: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            return {
                "title": "제목 생성 오류",
                "recommendations": ["", "", ""]
            }

    @classmethod
    def hybrid_generate_story(cls, data):
        """하이브리드 방식으로 이야기 생성 및 스트리밍"""
        try:
            # 스트림 시작을 알림
            yield "data: {\"status\": \"generating\"}\n\n"
            
            # 1. 파인튜닝된 모델로 기본 스토리 생성 (Chat Completion API 사용)
            prompt = cls._format_hybrid_prompt(data)
            print(f"\n[DEBUG] Fine-tuned model prompt: {json.dumps(prompt, ensure_ascii=False)}", flush=True)
            
            base_story = cls._generate_with_fine_tuned_model(prompt)
            print(f"\n[DEBUG] Fine-tuned model response: {base_story}", flush=True)
            
            if not base_story:
                raise Exception("기본 스토리 생성 실패")
                
            # 중간 결과 전달
            yield f"data: {json.dumps({'msg': 'base_story_completed', 'content': base_story}, ensure_ascii=False)}\n\n"
            
            # 2. GPT-4o로 스토리 확장 (Assistant API 사용)
            print("\n[DEBUG] Starting GPT-4o expansion...", flush=True)
            expanded_story = ""
            for content in cls._expand_with_gpt4o(base_story):
                expanded_story += content
                yield f"data: {json.dumps({'msg': 'expanding', 'content': content}, ensure_ascii=False)}\n\n"
                print(".", end="", flush=True)  # 진행 상황 표시
            
            print(f"\n[DEBUG] Final expanded story: {expanded_story}", flush=True)
            
            if not expanded_story:
                raise Exception("스토리 확장 실패")
            
            # 3. 제목 생성과 추천 생성을 한 번에 처리 (Chat Completion API 사용)
            title_and_recommendations = cls.generate_title_and_recommendations(
                expanded_story, 
                data.get('user_input', '')
            )
            
            title = title_and_recommendations["title"]
            recommendations = title_and_recommendations["recommendations"]
            
            print(f"\n[DEBUG] Generated title: {title}", flush=True)
            print(f"\n[DEBUG] Generated recommendations: {recommendations}", flush=True)
            
            # 최종 결과 포맷팅
            result = {
                "created_title": title,
                "created_content": expanded_story,
                "recommendations": recommendations
            }
            
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            print(f"[ERROR] Failed in hybrid story generation: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    @classmethod
    def hybrid_generate_story_with_assistant(cls, data):
        """하이브리드 방식으로 이야기 생성 및 스트리밍 (Assistant API 사용)"""
        try:
            # 사용자 ID는 이미 routes.py에서 설정됨
            user_id = data.get('user_id')
            if not user_id:
                raise Exception("유효한 사용자 ID가 없습니다.")
            
            # 기존 쓰레드 ID가 있으면 가져옴
            existing_thread_id = data.get('openai_thread_id')
            
            # 스트림 시작을 알림
            yield "data: {\"status\": \"generating\"}\n\n"
            
            # 1. 파인튜닝된 모델로 기본 스토리 생성
            prompt = cls._format_hybrid_prompt(data)
            print(f"\n[DEBUG] Fine-tuned model prompt: {json.dumps(prompt, ensure_ascii=False)}", flush=True)
            
            base_story = cls._generate_with_fine_tuned_model(prompt)
            print(f"\n[DEBUG] Fine-tuned model response: {base_story}", flush=True)
            
            if not base_story:
                raise Exception("기본 스토리 생성 실패")
                
            # 중간 결과 전달
            yield f"data: {json.dumps({'msg': 'base_story_completed', 'content': base_story}, ensure_ascii=False)}\n\n"
            
            # 2. GPT-4o로 스토리 확장 (스트리밍)
            print("\n[DEBUG] Starting GPT-4o expansion with user_id:", user_id, flush=True)
            expanded_story = ""
            
            # DB에서 이미 저장된 스레드 ID 조회
            db_thread_id = None
            if existing_thread_id:
                try:
                    with Database() as cursor:
                        cursor.execute(
                            """SELECT id FROM threads 
                               WHERE thread_id = %s LIMIT 1""",
                            (existing_thread_id,)
                        )
                        result = cursor.fetchone()
                        if result:
                            db_thread_id = result['id']
                            print(f"[DEBUG] Found existing DB thread_id: {db_thread_id}")
                except Exception as e:
                    print(f"[ERROR] Error querying DB thread: {str(e)}")
            
            for content in cls._expand_with_gpt4o(base_story, user_id, existing_thread_id):
                expanded_story = content  # 전체 내용을 한 번에 받음
                yield f"data: {json.dumps({'msg': 'expanding', 'content': expanded_story}, ensure_ascii=False)}\n\n"
                print(".", end="", flush=True)  # 진행 상황 표시
            
            print(f"\n[DEBUG] Final expanded story: {expanded_story}", flush=True)
            
            if not expanded_story:
                raise Exception("스토리 확장 실패")
            
            # 3. 제목 생성과 추천 생성을 한 번에 처리 (Chat Completion API 사용)
            title_and_recommendations = cls.generate_title_and_recommendations(
                expanded_story, 
                data.get('user_input', '')
            )
            
            title = title_and_recommendations["title"]
            recommendations = title_and_recommendations["recommendations"]
            
            print(f"\n[DEBUG] Generated title: {title}", flush=True)
            print(f"\n[DEBUG] Generated recommendations: {recommendations}", flush=True)
            
            # 최종 결과 포맷팅
            result = {
                "created_title": title,
                "created_content": expanded_story,
                "recommendations": recommendations
            }
            
            # DB에 저장
            thread_id = None
            conversation_id = None
            
            if db_thread_id:
                # 기존 스레드에 대화 추가
                conversation_id = cls.add_conversation_to_thread(db_thread_id, user_id, data, result)
                thread_id = db_thread_id
            else:
                # 새 스레드 생성
                thread_id, conversation_id = cls.save_to_database(user_id, data, result)
            
            if thread_id and conversation_id:
                result["thread_id"] = thread_id
                result["conversation_id"] = conversation_id
                result["user_id"] = user_id
                result["openai_thread_id"] = cls.current_openai_thread_id  # 클라이언트에 OpenAI 쓰레드 ID 전달
            
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            print(f"[ERROR] Failed in hybrid story generation: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"