from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from app.config import Config
from app.utils.database import Database
import json
from mysql.connector import Error
import requests
from app.utils.helpers import generate_session_hash
import traceback

class StoryService:
    client = OpenAI()

    @classmethod
    def generate_content(cls, prompt):
        """
        VAIV API를 사용하여 이야기를 생성하고 스트림으로 반환합니다.
        """
        print(f"[DEBUG] Starting generation with prompt: {prompt}")
        session_hash = generate_session_hash()
        print(f"[DEBUG] Generated session hash: {session_hash}")
        
        # 첫 번째 API 요청 - 큐 참여
        join_url = 'https://story.vaiv.kr/gradio_api/queue/join'
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
        response = requests.post(join_url, headers=headers, json=payload)
        print(f"[DEBUG] Join response status: {response.status_code}")
        print(f"[DEBUG] Join response content: {response.text}")
        
        if response.status_code != 200:
            raise Exception("API 요청 실패")
        
        # 두 번째 API 요청 - 데이터 가져오기
        data_url = f'https://story.vaiv.kr/gradio_api/queue/data?session_hash={session_hash}'
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
                                if data.get('msg') == 'process_completed':
                                    print("[DEBUG] Process completed")
                                    break
                                    
                                if data.get('output') and data['output'].get('data'):
                                    content = data['output']['data'][0]
                                    is_generating = data['output'].get('is_generating', True)
                                    print(f"[DEBUG] Content: {content}, is_generating: {is_generating}")
                                    
                                    if content and not is_generating:
                                        print(f"[DEBUG] Yielding content: {content}")
                                        yield content
                                        
                        except json.JSONDecodeError as e:
                            print(f"[DEBUG] JSON decode error: {e}")
                            continue

    @classmethod
    def generate_title(cls, content):
        """제목 생성"""
        completion = cls.client.chat.completions.create(
            model=Config.GPT_MODEL,
            messages=[
                {"role": "system", "content": "다음 고전소설 단락에 어울리는 **간략한 제목**을 생성해. **제목은 절대로 13글자를 넘기지 마.**"},
                {"role": "user", "content": content}
            ]
        )
        return completion.choices[0].message.content

    @classmethod
    def generate_recommendation(cls, theme):
        """추천 이야기 생성"""
        completion = cls.client.chat.completions.create(
            model=Config.GPT_MODEL,
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
    def save_to_database(cls, user_id, data, formatted_data):
        """생성된 결과를 데이터베이스에 저장"""
        connection = Database.get_connection()
        if not connection:
            return None, None
            
        try:
            print(f"[INFO] Saving story to database for user_id: {user_id}")
            cursor = connection.cursor()
            
            # thread_id가 없거나 null이거나 빈 문자열이면 새 thread 생성
            thread_id = data.get('thread_id')  # None if key doesn't exist or value is null
            
            if thread_id:  # thread_id가 있는 경우 유효성 검사
                try:
                    # 정수로 변환 시도
                    thread_id = int(thread_id)
                    
                    # thread가 존재하고 해당 user의 것인지 확인
                    cursor.execute(
                        "SELECT 1 FROM threads WHERE thread_id = %s AND user_id = %s",
                        (thread_id, user_id)
                    )
                    if not cursor.fetchone():
                        # 유효하지 않은 thread_id면 새로 생성
                        thread_id = None
                        
                except (ValueError, TypeError):
                    # 정수로 변환 실패하면 새로 생성
                    thread_id = None
            
            if not thread_id:  # 새 thread 생성
                # 첫 번째 conversation의 제목을 thread title로 설정
                cursor.execute(
                    "INSERT INTO threads (user_id, title) VALUES (%s, %s)",
                    (user_id, formatted_data.get('created_title', ''))
                )
                thread_id = cursor.lastrowid
            else:
                # 기존 thread의 updated_at 업데이트
                cursor.execute(
                    "UPDATE threads SET updated_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                    (thread_id,)
                )
            
            # 해당 thread의 마지막 conversation_id 조회
            cursor.execute(
                """SELECT MAX(conversation_id) 
                   FROM conversations 
                   WHERE thread_id = %s""",
                (thread_id,)
            )
            max_conversation_id = cursor.fetchone()[0]
            conversation_id = (max_conversation_id or 0) + 1
            
            # conversation 생성
            cursor.execute(
                """INSERT INTO conversations 
                   (conversation_id, thread_id) 
                   VALUES (%s, %s)""",
                (conversation_id, thread_id)
            )
            
            # conversation_data 저장
            data_entries = [
                ('user_input', formatted_data.get('user_input', '')),
                ('tags', json.dumps(formatted_data.get('tags', {}), ensure_ascii=False)),
                ('created_title', formatted_data.get('created_title', '')),
                ('created_content', formatted_data.get('created_content', '')),
                ('similar_1', '{}'),
                ('similar_2', '{}'),
                ('similar_3', '{}'),
                ('recommended_1', ''),
                ('recommended_2', ''),
                ('recommended_3', '')
            ]
            
            # 각 데이터 항목을 개별적으로 저장
            for category, value in data_entries:
                try:
                    cursor.execute(
                        """INSERT INTO conversation_data 
                           (conversation_id, thread_id, category, data) 
                           VALUES (%s, %s, %s, %s)""",
                        (conversation_id, thread_id, category, value)
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to insert {category}: {str(e)}")
                    raise
            
            connection.commit()
            print(f"[INFO] Successfully saved story. thread_id: {thread_id}, conversation_id: {conversation_id}")
            return thread_id, conversation_id
            
        except Exception as e:
            print(f"[ERROR] Failed to save to database: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            if connection:
                connection.rollback()
            return None, None
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

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
            title = cls.generate_title(content)
            
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
    def update_search_results(cls, thread_id: int, conversation_id: int, user_id: int, 
                             search_results: list, recommendations: list) -> dict:
        """검색 결과와 추천 문서를 DB에 업데이트"""
        connection = Database.get_connection()
        if not connection:
            return {"success": False, "error": "Database connection failed"}
        
        try:
            cursor = connection.cursor()
            
            # 소유권 확인
            cursor.execute(
                """SELECT 1 FROM threads t 
                   JOIN conversations c ON t.thread_id = c.thread_id 
                   WHERE t.thread_id = %s AND t.user_id = %s AND c.conversation_id = %s""",
                (thread_id, user_id, conversation_id)
            )
            if not cursor.fetchone():
                return {"success": False, "error": "Invalid thread_id, conversation_id, or unauthorized"}
            
            # 검색 결과 업데이트
            for idx, (category, result) in enumerate([
                ('similar_1', search_results[0] if len(search_results) > 0 else {}),
                ('similar_2', search_results[1] if len(search_results) > 1 else {}),
                ('similar_3', search_results[2] if len(search_results) > 2 else {})
            ], 1):
                cursor.execute(
                    """UPDATE conversation_data 
                       SET data = %s 
                       WHERE thread_id = %s AND conversation_id = %s AND category = %s""",
                    (json.dumps(result, ensure_ascii=False), thread_id, conversation_id, category)
                )
            
            # 추천 문서 업데이트
            for idx, (category, recommendation) in enumerate([
                ('recommended_1', recommendations[0] if len(recommendations) > 0 else ""),
                ('recommended_2', recommendations[1] if len(recommendations) > 1 else ""),
                ('recommended_3', recommendations[2] if len(recommendations) > 2 else "")
            ], 1):
                cursor.execute(
                    """UPDATE conversation_data 
                       SET data = %s 
                       WHERE thread_id = %s AND conversation_id = %s AND category = %s""",
                    (recommendation, thread_id, conversation_id, category)
                )
            
            connection.commit()
            return {"success": True}
            
        except Exception as e:
            print(f"Error updating search results: {str(e)}")
            if connection:
                connection.rollback()
            return {"success": False, "error": str(e)}
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()