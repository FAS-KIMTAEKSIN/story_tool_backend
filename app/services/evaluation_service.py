from app.utils.database import Database

class EvaluationService:
    @classmethod
    def update_content_evaluation(cls, thread_id: int, conversation_id: int, user_id: int, evaluation: str) -> dict:
        """콘텐츠 평가를 업데이트하거나 생성합니다"""
        try:
            with Database() as db:
                # 기존 평가 확인
                check_query = """
                    SELECT evaluation 
                    FROM content_evaluation 
                    WHERE thread_id = %s 
                    AND conversation_id = %s 
                    AND user_id = %s
                """
                db.execute(check_query, (thread_id, conversation_id, user_id))
                existing_evaluation = db.fetchone()

                if existing_evaluation:
                    if existing_evaluation['evaluation'] == evaluation:
                        # 같은 평가가 이미 있으면 삭제 (토글)
                        delete_query = """
                            DELETE FROM content_evaluation 
                            WHERE thread_id = %s 
                            AND conversation_id = %s 
                            AND user_id = %s
                        """
                        db.execute(delete_query, (thread_id, conversation_id, user_id))
                        return {"success": True, "action": "removed"}
                    else:
                        # 다른 평가가 있으면 업데이트
                        update_query = """
                            UPDATE content_evaluation 
                            SET evaluation = %s 
                            WHERE thread_id = %s 
                            AND conversation_id = %s 
                            AND user_id = %s
                        """
                        db.execute(update_query, (evaluation, thread_id, conversation_id, user_id))
                        return {"success": True, "action": "updated"}
                else:
                    # 새로운 평가 추가
                    insert_query = """
                        INSERT INTO content_evaluation 
                        (thread_id, conversation_id, user_id, evaluation) 
                        VALUES (%s, %s, %s, %s)
                    """
                    db.execute(insert_query, (thread_id, conversation_id, user_id, evaluation))
                    return {"success": True, "action": "created"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    @classmethod
    def get_content_evaluation(cls, thread_id: int, conversation_id: int, user_id: int) -> dict:
        """특정 콘텐츠에 대한 사용자의 평가를 조회합니다"""
        try:
            with Database() as db:
                query = """
                    SELECT evaluation 
                    FROM content_evaluation 
                    WHERE thread_id = %s 
                    AND conversation_id = %s 
                    AND user_id = %s
                """
                db.execute(query, (thread_id, conversation_id, user_id))
                result = db.fetchone()
                
                return {
                    "success": True,
                    "evaluation": result['evaluation'] if result else None
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)} 