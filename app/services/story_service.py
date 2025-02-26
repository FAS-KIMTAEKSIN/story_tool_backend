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
    def generate_title(cls, content):
        """제목 생성"""
        completion = cls.client.chat.completions.create(
            model=Config.GPT_MODEL,
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
        def transaction_callback(cursor):
            try:
                thread_id = data.get('thread_id')
                
                if thread_id:  # thread_id가 있는 경우 유효성 검사
                    try:
                        thread_id = int(thread_id)
                        cursor.execute(
                            """SELECT 1 FROM threads 
                               WHERE thread_id = %s AND user_id = %s 
                               FOR UPDATE""",
                            (thread_id, user_id)
                        )
                        if not cursor.fetchone():
                            thread_id = None
                    except (ValueError, TypeError):
                        thread_id = None
                
                if not thread_id:  # 새 thread 생성
                    thread_id = Database.get_next_thread_id(cursor, user_id)
                    cursor.execute(
                        """INSERT INTO threads (thread_id, user_id, title) 
                           VALUES (%s, %s, %s)""",
                        (thread_id, user_id, formatted_data.get('created_title', ''))
                    )
                else:
                    # 기존 thread의 updated_at 업데이트
                    cursor.execute(
                        """UPDATE threads 
                           SET updated_at = CURRENT_TIMESTAMP 
                           WHERE thread_id = %s""",
                        (thread_id,)
                    )
                
                # 새로운 conversation_id 생성
                conversation_id = Database.get_next_conversation_id(cursor, thread_id)
                
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
                
                for category, value in data_entries:
                    cursor.execute(
                        """INSERT INTO conversation_data 
                           (conversation_id, thread_id, category, data) 
                           VALUES (%s, %s, %s, %s)""",
                        (conversation_id, thread_id, category, value)
                    )
                
                return thread_id, conversation_id
            except Exception as e:
                print(f"[ERROR] Transaction callback error: {str(e)}")
                raise  # 상위 트랜잭션 핸들러로 예외 전파

        try:
            return Database.execute_transaction(transaction_callback)
        except Exception as e:
            print(f"[ERROR] Failed to save to database: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            return None, None

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

    @classmethod
    def _expand_with_gpt4o(cls, base_story):
        """GPT-4o로 스토리 확장 (스트리밍)"""
        try:
            print("[DEBUG] Creating GPT-4o completion request...")
            expanded_story = ""
            buffer = ""  # 버퍼 추가
            
            response = cls.client.chat.completions.create(
                model=Config.GPT_MODEL,
                messages=[
                    {"role": "system", "content": Config.HYBRID_SYSTEM_PROMPT},
                    {"role": "user", "content": f"다음 이야기를 더 풍부하게 확장해주세요:\n\n{base_story}"}
                ],
                temperature=0.7,
                max_tokens=2048,
                stream=True
            )
            print("[DEBUG] GPT-4o request created, starting stream...")
            
            for chunk in response:
                try:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        buffer += content  # 버퍼에 추가
                        expanded_story += content
                        
                        # 버퍼가 더 큰 크기(100자)가 되거나 문장이 끝나면 yield
                        if len(buffer) >= 100 or any(buffer.endswith(end) for end in ['.', '!', '?', '\n']):
                            yield buffer
                            buffer = ""  # 버퍼 초기화
                except Exception as e:
                    print(f"[ERROR] Error processing chunk: {str(e)}")
                    if buffer:  # 오류 발생 시 버퍼 내용 전송
                        yield buffer
                        buffer = ""
                    continue
            
            # 남은 버퍼가 있으면 전송
            if buffer:
                yield buffer
            
            print(f"[DEBUG] Stream completed. Total length: {len(expanded_story)}")
            return expanded_story
            
        except Exception as e:
            print(f"[ERROR] Error in GPT-4o expansion: {str(e)}")
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            return None

    @classmethod
    def hybrid_generate_story(cls, data):
        """하이브리드 방식으로 이야기 생성 및 스트리밍"""
        try:
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
            print("\n[DEBUG] Starting GPT-4o expansion...", flush=True)
            expanded_story = ""
            for content in cls._expand_with_gpt4o(base_story):
                expanded_story += content
                yield f"data: {json.dumps({'msg': 'expanding', 'content': content}, ensure_ascii=False)}\n\n"
                print(".", end="", flush=True)  # 진행 상황 표시
            
            print(f"\n[DEBUG] Final expanded story: {expanded_story}", flush=True)
            
            if not expanded_story:
                raise Exception("스토리 확장 실패")
            
            # 제목 생성
            title = cls.generate_title(expanded_story)
            print(f"\n[DEBUG] Generated title: {title}", flush=True)
            
            # 최종 결과 포맷팅
            result = {
                "created_title": title,
                "created_content": expanded_story
            }
            
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            print(f"[ERROR] Failed in hybrid story generation: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

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
        """파인튜닝된 모델로 스토리 생성"""
        try:
            response = cls.client.chat.completions.create(
                model=Config.FINE_TUNED_MODEL,
                messages=prompt["messages"],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Error in fine-tuned generation: {str(e)}")
            return None