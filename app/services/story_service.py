from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from app.config import Config
from app.utils.database import Database
import json
from mysql.connector import Error

class StoryService:
    client = OpenAI()

    @classmethod
    def generate_content(cls, prompt):
        """본문 생성"""
        completion = cls.client.chat.completions.create(
            model=Config.FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": "당신은 다양한 장르의 전문 작가입니다. 제시된 라벨링 기준을 참고하여 이야기를 생성해 주세요.\n\n- 주어진 내용분류, 주제어, 주제문은 생성될 단락의 라벨링입니다.\n- 장르와 배경에 맞는 적절한 문체와 표현을 사용해주세요.\n- 인물의 행동, 대화, 감정을 자연스럽게 표현해주세요.\n- 상황과 인물 관계를 효과적으로 담아주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content

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
        """스토리 생성"""
        from app.services.search_service import SearchService

        prompt = SearchService.process_input(data)
        created_content = cls.generate_content(prompt)

        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            title_future = executor.submit(cls.generate_title, created_content)
            recommendation_futures = [
                executor.submit(cls.generate_recommendation, data.get('user_input', ''))
                for _ in range(3)
            ]

            created_title = title_future.result()
            recommendations = [f.result() for f in recommendation_futures]

        return {
            "created_title": created_title,
            "created_content": created_content,
            "recommendations": recommendations
        }