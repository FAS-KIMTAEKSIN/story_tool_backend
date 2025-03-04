from app.utils.database import Database
from mysql.connector import Error
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class HistoryService:
    @staticmethod
    def get_preview_text(text: str, search_text: str, context_length: int = 100) -> str:
        """검색어가 포함된 부분의 앞뒤 문맥을 포함한 미리보기 텍스트를 생성합니다"""
        if not text or not search_text:
            return ""
            
        # 검색어 위치 찾기
        start_pos = text.lower().find(search_text.lower())
        if start_pos == -1:
            return text[:200] + "..." if len(text) > 200 else text
            
        # 앞뒤 문맥 범위 계산
        preview_start = max(0, start_pos - context_length)
        preview_end = min(len(text), start_pos + len(search_text) + context_length)
        
        # 미리보기 텍스트 생성
        preview = ""
        if preview_start > 0:
            preview += "..."
        preview += text[preview_start:preview_end]
        if preview_end < len(text):
            preview += "..."
            
        return preview

    @staticmethod
    def format_json_preview(json_data: dict) -> str:
        """JSON 데이터를 읽기 쉬운 형식으로 변환합니다"""
        try:
            # 태그 데이터 특별 처리
            if 'tags' in json_data:
                return ', '.join(str(tag) for tag in json_data.get('tags', []))
            # 유사 문서 데이터 특별 처리
            elif 'title' in json_data and 'content' in json_data:
                return f"{json_data.get('title', '')}: {json_data.get('content', '')}"
            # 기타 JSON 데이터는 값만 추출하여 문자열로 결합
            else:
                return ' '.join(str(v) for v in json_data.values())
        except Exception:
            return ""

    @staticmethod
    def format_search_result(row: Dict[str, Any], search_text: str) -> Dict[str, Any]:
        """검색 결과 행을 포맷팅합니다"""
        preview = None
        if row['matched_data']:
            matched_items = row['matched_data'].split('||')
            
            # 검색 우선순위 정의
            priority_order = ['user_input', 'created_title', 'created_content', 
                             'similar_1', 'similar_2', 'similar_3']
            
            # 우선순위에 따라 정렬된 결과를 저장할 딕셔너리
            category_results = {}
            
            for item in matched_items:
                if not item:
                    continue
                try:
                    category, content = item.split(':', 1)
                    
                    # JSON 데이터 처리
                    if category in ['similar_1', 'similar_2', 'similar_3']:
                        try:
                            json_data = json.loads(content)
                            content = HistoryService.format_json_preview(json_data)
                            if not content:
                                continue
                        except (json.JSONDecodeError, AttributeError):
                            continue
                    
                    preview_text = HistoryService.get_preview_text(content, search_text)
                    if preview_text:
                        category_results[category] = {
                            "category": {
                                'user_input': '사용자 입력',
                                'created_title': '제목',
                                'created_content': '내용',
                                'similar_1': '유사한 고전 원문',
                                'similar_2': '유사한 고전 원문',
                                'similar_3': '유사한 고전 원문'
                            }.get(category, category),
                            "text": preview_text
                        }
                except (ValueError, AttributeError):
                    continue
            
            # 우선순위에 따라 첫 번째 매칭된 결과 선택
            for category in priority_order:
                if category in category_results:
                    preview = category_results[category]
                    break

        return {
            "thread_id": row['thread_id'],
            "conversation_id": row['conversation_id'],
            "thread_created_at": row['thread_created_at'].isoformat() if row['thread_created_at'] else None,
            "thread_updated_at": row['thread_updated_at'].isoformat() if row['thread_updated_at'] else None,
            "title": row['title'] or "",
            "preview": preview,
            "user_input": row['user_input'] or ""
        }

    @staticmethod
    def create_empty_conversation() -> Dict[str, Any]:
        """빈 대화 데이터 구조를 생성합니다"""
        return {
            'user_input': '',
            'tags': {},
            'created_title': '',
            'created_content': '',
            'similar_1': {},
            'similar_2': {},
            'similar_3': {},
            'recommended_1': '',
            'recommended_2': '',
            'recommended_3': ''
        }

    @staticmethod
    def process_conversation_data(conversations: Dict[int, Dict], row: Dict[str, Any]) -> None:
        """대화 데이터를 처리하여 conversations 딕셔너리를 업데이트합니다"""
        conv_id = row['conversation_id']
        if conv_id not in conversations:
            conversations[conv_id] = HistoryService.create_empty_conversation()
            conversations[conv_id]['conversation_id'] = conv_id

        category = row['category']
        data = row['data']

        if category in ['tags', 'similar_1', 'similar_2', 'similar_3']:
            conversations[conv_id][category] = json.loads(data) if data else {}
        else:
            conversations[conv_id][category] = data

    @classmethod
    def verify_thread_ownership(cls, cursor, thread_id: int, user_id: int) -> bool:
        """스레드의 소유권을 확인합니다"""
        cursor.execute(
            "SELECT 1 FROM threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id)
        )
        return bool(cursor.fetchone())

    @classmethod
    def search_history(cls, user_id: int, search_text: str) -> dict:
        """사용자의 대화 기록에서 검색어가 포함된 내용을 찾습니다"""
        connection = Database.get_connection()
        if not connection:
            return {"success": False, "error": "Database connection failed"}

        try:
            cursor = connection.cursor(dictionary=True)
            
            search_pattern = f"%{search_text}%"
            # JSON 검색을 위한 패턴은 % 없이 전달
            cursor.execute(cls._get_search_query(), 
                          (search_text, search_pattern, user_id, search_text, search_pattern))
            results = cursor.fetchall()
            
            search_results = [cls.format_search_result(row, search_text) 
                            for row in results]
            
            return {
                "success": True,
                "results": search_results,
                "total_count": len(search_results)
            }
            
        except Exception as e:
            print(f"Error searching history: {str(e)}")
            return {"success": False, "error": str(e)}
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    @classmethod
    def get_chat_history_detail(cls, thread_id: int, user_id: int) -> dict:
        """특정 thread의 대화 기록을 상세히 조회합니다"""
        connection = Database.get_connection()
        if not connection:
            return {"success": False, "error": "Database connection failed"}

        try:
            cursor = connection.cursor(dictionary=True)
            
            if not cls.verify_thread_ownership(cursor, thread_id, user_id):
                return {"success": False, "error": "Invalid thread_id or user_id"}
            
            cursor.execute(cls._get_conversation_query(), (thread_id,))
            rows = cursor.fetchall()
            
            if not rows:
                return {"success": False, "error": "No conversations found"}
            
            conversations = {}
            for row in rows:
                cls.process_conversation_data(conversations, row)
            
            conversation_list = list(conversations.values())
            
            return {
                "success": True,
                "conversation_history": [
                    {
                        "success": True,
                        "result": conv,
                        "thread_id": thread_id,
                        "conversation_id": conv['conversation_id'],
                        "user_id": user_id
                    } for conv in conversation_list
                ]
            }
            
        except Error as e:
            print(f"Database error: {str(e)}")
            return {"success": False, "error": f"Database error: {str(e)}"}
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def _get_search_query() -> str:
        """검색 쿼리를 반환합니다"""
        return """
            SELECT DISTINCT 
                t.thread_id,
                t.created_at as thread_created_at,
                t.updated_at as thread_updated_at,
                c.conversation_id,
                MAX(CASE WHEN cd.category = 'user_input' THEN cd.data END) as user_input,
                MAX(CASE WHEN cd.category = 'created_title' THEN cd.data END) as title,
                MAX(CASE WHEN cd.category = 'created_content' THEN cd.data END) as content,
                GROUP_CONCAT(
                    CASE 
                        WHEN cd.category IN ('similar_1', 'similar_2', 'similar_3') THEN
                            CASE 
                                WHEN JSON_VALID(cd.data) AND 
                                     JSON_SEARCH(JSON_EXTRACT(cd.data, '$.*'), 'one', %s) IS NOT NULL 
                                THEN CONCAT(cd.category, ':', cd.data)
                            END
                        WHEN cd.category IN ('user_input', 'created_title', 'created_content')
                            AND cd.data LIKE %s THEN CONCAT(cd.category, ':', cd.data)
                    END SEPARATOR '||'
                ) as matched_data
            FROM threads t
            JOIN conversations c ON t.thread_id = c.thread_id
            JOIN conversation_data cd ON c.thread_id = cd.thread_id 
                AND c.conversation_id = cd.conversation_id
            WHERE t.user_id = %s
            AND (
                (cd.category IN ('similar_1', 'similar_2', 'similar_3') 
                 AND JSON_VALID(cd.data) 
                 AND JSON_SEARCH(JSON_EXTRACT(cd.data, '$.*'), 'one', %s) IS NOT NULL)
                OR 
                (cd.category IN ('user_input', 'created_title', 'created_content')
                 AND cd.data LIKE %s)
            )
            GROUP BY t.thread_id, c.conversation_id
            ORDER BY t.updated_at DESC, c.conversation_id DESC
        """

    @staticmethod
    def _get_conversation_query() -> str:
        """대화 조회 쿼리를 반환합니다"""
        return """
            SELECT 
                c.conversation_id,
                cd.category,
                cd.data
            FROM conversations c
            JOIN conversation_data cd 
                ON c.thread_id = cd.thread_id 
                AND c.conversation_id = cd.conversation_id
            JOIN threads t ON c.thread_id = t.thread_id
            WHERE c.thread_id = %s
            ORDER BY c.conversation_id ASC, cd.category
        """

    @classmethod
    def get_chat_history_list(cls, user_id: int) -> dict:
        """사용자의 전체 대화 목록을 조회합니다"""
        connection = Database.get_connection()
        if not connection:
            return {"success": False, "error": "Database connection failed"}

        try:
            cursor = connection.cursor(dictionary=True)
            
            # 사용자의 모든 대화 목록 조회
            cursor.execute("""
                SELECT 
                    t.thread_id,
                    t.title,
                    t.created_at as thread_created_at,
                    t.updated_at as thread_updated_at,
                    COALESCE(c.conversation_id, 1) as conversation_id,
                    MAX(CASE WHEN cd.category = 'user_input' THEN cd.data END) as user_input,
                    MAX(CASE WHEN cd.category = 'created_content' THEN cd.data END) as content
                FROM threads t
                LEFT JOIN conversations c ON t.thread_id = c.thread_id AND c.conversation_id = 1
                LEFT JOIN conversation_data cd ON c.thread_id = cd.thread_id 
                    AND c.conversation_id = cd.conversation_id
                WHERE t.user_id = %s
                GROUP BY t.thread_id, t.title, t.created_at, t.updated_at, c.conversation_id
                ORDER BY COALESCE(t.updated_at, t.created_at) DESC, t.thread_id DESC
            """, (user_id,))
            
            rows = cursor.fetchall()
            
            # 결과 포맷팅
            chat_history = []
            for row in rows:
                chat_history.append({
                    "thread_id": row['thread_id'],
                    "conversation_id": row['conversation_id'],
                    "thread_created_at": row['thread_created_at'].isoformat() if row['thread_created_at'] else None,
                    "thread_updated_at": row['thread_updated_at'].isoformat() if row['thread_updated_at'] else None,
                    "title": row['title'] or "",
                    "preview": {
                        "user_input": row['user_input'] or "",
                        "content": row['content'][:200] + "..." if row['content'] and len(row['content']) > 200 else row['content'] or ""
                    }
                })
            
            return {
                "success": True,
                "chat_history": chat_history,
                "total_count": len(chat_history)
            }
            
        except Error as e:
            print(f"Database error: {str(e)}")
            return {"success": False, "error": f"Database error: {str(e)}"}
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    @classmethod
    def delete_thread(cls, thread_id: int, user_id: int) -> dict:
        """사용자의 대화 쓰레드를 삭제합니다"""
        connection = Database.get_connection()
        if not connection:
            return {"success": False, "error": "Database connection failed"}

        try:
            cursor = connection.cursor(dictionary=True)
            
            # 쓰레드 소유권 확인
            if not cls.verify_thread_ownership(cursor, thread_id, user_id):
                return {"success": False, "error": "Invalid thread_id or unauthorized"}
            
            # 쓰레드 삭제 (CASCADE로 인해 관련된 conversations와 conversation_data도 자동 삭제됨)
            cursor.execute(
                "DELETE FROM threads WHERE thread_id = %s AND user_id = %s",
                (thread_id, user_id)
            )
            
            if cursor.rowcount == 0:
                return {"success": False, "error": "Thread not found or already deleted"}
            
            connection.commit()
            return {"success": True}
            
        except Error as e:
            print(f"Database error: {str(e)}")
            if connection:
                connection.rollback()
            return {"success": False, "error": f"Database error: {str(e)}"}
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    @classmethod
    def update_thread_title(cls, thread_id: int, user_id: int, title: str) -> dict:
        """쓰레드의 제목을 업데이트합니다"""
        if not title or not title.strip():
            return {"success": False, "error": "제목이 비어있습니다"}

        connection = Database.get_connection()
        if not connection:
            return {"success": False, "error": "Database connection failed"}

        try:
            cursor = connection.cursor(dictionary=True)
            
            # 쓰레드 소유권 확인
            if not cls.verify_thread_ownership(cursor, thread_id, user_id):
                return {"success": False, "error": "Invalid thread_id or unauthorized"}
            
            # 제목 업데이트
            cursor.execute(
                "UPDATE threads SET title = %s WHERE thread_id = %s AND user_id = %s",
                (title.strip(), thread_id, user_id)
            )
            
            if cursor.rowcount == 0:
                return {"success": False, "error": "Thread not found or no changes made"}
            
            connection.commit()
            return {"success": True}
            
        except Error as e:
            print(f"Database error: {str(e)}")
            if connection:
                connection.rollback()
            return {"success": False, "error": f"Database error: {str(e)}"}
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close() 