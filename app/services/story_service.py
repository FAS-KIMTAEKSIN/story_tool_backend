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
                ('similar_1', json.dumps(formatted_data.get('similar_1', {}), ensure_ascii=False)),
                ('similar_2', json.dumps(formatted_data.get('similar_2', {}), ensure_ascii=False)),
                ('similar_3', json.dumps(formatted_data.get('similar_3', {}), ensure_ascii=False)),
                ('recommended_1', formatted_data.get('recommended_1', '')),
                ('recommended_2', formatted_data.get('recommended_2', '')),
                ('recommended_3', formatted_data.get('recommended_3', ''))
            ]
            
            # 각 데이터 항목을 개별적으로 저장
            for category, value in data_entries:
                cursor.execute(
                    """INSERT INTO conversation_data 
                       (conversation_id, thread_id, category, data) 
                       VALUES (%s, %s, %s, %s)""",
                    (conversation_id, thread_id, category, value)
                )
            
            connection.commit()
            return thread_id, conversation_id
            
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            if connection:
                connection.rollback()
            return None, None
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    @classmethod
    def generate_story(cls, data):
        try:
            from app.services.search_service import SearchService
            
            # 1. 프롬프트 생성
            prompt = SearchService.process_input(data)
            
            # 2. 이야기 생성
            url = Config.VAIV_QUEUE_JOIN_URL
            headers = {
                'Accept': '*/*',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive',
                'Content-Type': 'application/json'
            }
            
            session_hash = generate_session_hash()
            payload = {
                "data": [prompt],
                "event_data": None,
                "fn_index": 0,
                "trigger_id": 4,
                "session_hash": session_hash
            }
            
            # 큐 참여
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                raise Exception(f"큐 참여 실패: {response.text}")
            
            # 데이터 가져오기
            data_url = f'{Config.VAIV_QUEUE_DATA_URL}?session_hash={session_hash}'
            headers = {
                'Accept': 'text/event-stream',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                'Connection': 'keep-alive'
            }
            
            content = None
            with requests.get(data_url, headers=headers, stream=True) as response:
                if response.status_code != 200:
                    raise Exception(f"데이터 가져오기 실패: {response.text}")
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            try:
                                data = json.loads(decoded_line[6:])
                                # 그라디오 응답을 그대로 전달
                                yield decoded_line[6:]  # 'data: ' 제외하고 전달
                                
                                # process_completed 메시지에서 최종 컨텐츠 저장
                                if data.get('msg') == 'process_completed':
                                    if data.get('output') and data['output'].get('data'):
                                        content = data['output']['data'][0]
                                        break
                                            
                            except json.JSONDecodeError:
                                continue
            
            if not content:
                raise Exception("이야기 생성 실패: 컨텐츠가 비어있음")

            # 3. 제목 생성
            title = cls.generate_title(content)
            
            # 4. 추천 이야기 생성
            recommendations = [
                cls.generate_recommendation(data.get('user_input', ''))
                for _ in range(3)
            ]
            
            # 최종 결과를 딕셔너리로 yield
            yield {
                "created_title": title,
                "created_content": content,
                "recommendations": recommendations
            }
            
        except Exception as e:
            print(f"[DEBUG] ERROR in generate_story: {str(e)}")
            print(f"[DEBUG] ERROR traceback: {traceback.format_exc()}")
            raise e