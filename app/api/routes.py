from flask import Blueprint, request, jsonify, Response, stream_with_context, current_app
from app.services.story_service import StoryService
from app.services.search_service import SearchService
from app.services.analysis_service import AnalysisService
from app.utils.decorators import log_request_response
from app.utils.database import Database
from mysql.connector import Error
from concurrent.futures import ThreadPoolExecutor
import json
from app.services.history_service import HistoryService
from app.services.evaluation_service import EvaluationService
import traceback
import time
from app.services.story_service import ConversationManager, ThreadManager

api_bp = Blueprint('api', __name__)

@api_bp.route('/search', methods=['POST'])
@log_request_response
def search_documents():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400

        required_fields = ['user_input', 'tags', 'thread_id', 'conversation_id', 'user_id']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "필수 필드가 누락되었습니다"}), 400
            
        # 검색 및 추천 수행
        results = SearchService.search_and_recommend(data)
        search_results = results['search_results']
        recommendations = results['recommendations']
        
        # DB 업데이트
        update_result = StoryService.update_search_results(
            thread_id=data['thread_id'],
            conversation_id=data['conversation_id'],
            user_id=data['user_id'],
            search_results=search_results,
            recommendations=recommendations
        )
        
        if not update_result['success']:
            return jsonify({"error": "DB 업데이트 실패"}), 500
        
        return jsonify({
            "success": True,
            "result": {
                "similar_1": search_results[0] if len(search_results) > 0 else {},
                "similar_2": search_results[1] if len(search_results) > 1 else {},
                "similar_3": search_results[2] if len(search_results) > 2 else {},
                "recommended_1": recommendations[0] if len(recommendations) > 0 else "",
                "recommended_2": recommendations[1] if len(recommendations) > 1 else "",
                "recommended_3": recommendations[2] if len(recommendations) > 2 else ""
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/generate', methods=['POST'])
@log_request_response
def generate_story():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400

        # 임시 사용자 ID 가져오기
        user_id = data.get('user_id') or Database.get_or_create_temp_user()
        if user_id is None:
            return jsonify({"error": "사용자를 처리할 수 없습니다"}), 500
            
        # 사용자 ID 설정
        data['user_id'] = user_id
            
        def generate():
            try:
                # thread_id가 없으면 새로 생성
                thread_id = data.get('thread_id')
                conversation_id = data.get('conversation_id')
                ids_created = False
                
                # thread_id가 없으면 새로 생성
                if not thread_id:
                    try:
                        thread_id = ThreadManager.create_thread(user_id)
                        data['thread_id'] = thread_id
                        current_app.logger.info(f"Created new thread: {thread_id}")
                        
                        # conversation_id 생성
                        conversation_id = ConversationManager.create_conversation(thread_id)
                        data['conversation_id'] = conversation_id
                        current_app.logger.info(f"Created new conversation: {conversation_id}")
                        ids_created = True
                        
                        # thread_id와 conversation_id를 한 번에 전달
                        yield f"data: {json.dumps({'thread_id': thread_id, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        current_app.logger.error(f"Failed to create thread or conversation: {str(e)}")
                        raise Exception("스레드 또는 대화 생성에 실패했습니다")
                # thread_id는 있지만 conversation_id가 없는 경우
                elif not conversation_id:
                    try:
                        conversation_id = ConversationManager.create_conversation(thread_id)
                        data['conversation_id'] = conversation_id
                        current_app.logger.info(f"Created new conversation: {conversation_id}")
                        ids_created = True
                        
                        # conversation_id 전달
                        yield f"data: {json.dumps({'thread_id': thread_id, 'conversation_id': conversation_id}, ensure_ascii=False)}\n\n"
                    except Exception as e:
                        current_app.logger.error(f"Failed to create conversation: {str(e)}")
                        raise Exception("대화 생성에 실패했습니다")
                
                # ID 생성 후 또는 이미 존재하는 ID 확인
                if not ids_created:
                    current_app.logger.info(f"Using existing thread_id: {thread_id} and conversation_id: {conversation_id}")
                
                # user_input과 tags 저장
                with Database() as db:
                    db.connection.start_transaction()
                    try:
                        ConversationManager.save_conversation_data(
                            db, thread_id, conversation_id, 
                            'user_input', data.get('user_input', '')
                        )
                        ConversationManager.save_conversation_data(
                            db, thread_id, conversation_id, 
                            'tags', json.dumps(data.get('tags', {}), ensure_ascii=False)
                        )
                        db.connection.commit()
                        current_app.logger.info(f"Saved initial data for conversation: {conversation_id}")
                    except Exception as e:
                        db.connection.rollback()
                        current_app.logger.error(f"Failed to save initial data: {str(e)}")
                        raise Exception("초기 데이터 저장에 실패했습니다")
                
                story_generator = StoryService.hybrid_generate_story_with_assistant(data)
                final_content = None
                generated_result = None
                recommendations = None
                openai_thread_id = None
                
                for message in story_generator:
                    yield message  # 이미 SSE 형식으로 포맷팅된 메시지
                    
                    # JSON 파싱하여 최종 결과 확인
                    try:
                        parsed = json.loads(message.replace('data: ', ''))
                        
                        # 생성된 대화 정보 캡처
                        if 'conversation_created' in parsed.get('msg', ''):
                            thread_id = parsed.get('thread_id')
                            conversation_id = parsed.get('conversation_id')
                            
                        # 최종 결과 캡처
                        if 'created_content' in parsed:
                            generated_result = parsed
                            final_content = parsed['created_content']
                            recommendations = parsed.get('recommendations', ["", "", ""])
                            openai_thread_id = parsed.get('openai_thread_id')
                            thread_id = parsed.get('thread_id', thread_id)
                            conversation_id = parsed.get('conversation_id', conversation_id)
                    except:
                        continue

                if not thread_id or not conversation_id:
                    raise Exception("대화 생성 실패: ID 정보가 없습니다")
                
                # 마지막 응답을 더 안정적으로 전송하기 위한 지연 및 분할
                time.sleep(0.5)  # 마지막 응답 전 더 긴 지연
                
                # 마지막 응답 전에 완료 신호 전송
                yield "data: {\"msg\": \"completion_pending\"}\n\n"
                time.sleep(0.2)
                
                # 최종 응답 전송
                final_result = {
                    "success": True,
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "openai_thread_id": openai_thread_id
                }
                
                final_json = json.dumps(final_result, ensure_ascii=False)
                yield f"data: {final_json}\n\n"
                
            except Exception as e:
                print(f"[ERROR] Failed to generate story: {str(e)}")
                print(f"[ERROR] {traceback.format_exc()}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/analyze', methods=['POST'])
@log_request_response
def analyze_work():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        result = AnalysisService.analyze_work(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/saveChatHistory', methods=['POST'])
@log_request_response
def save_chat_history():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # 임시 사용자 ID 가져오기
        user_id = Database.get_or_create_temp_user()
        if user_id is None:
            return jsonify({"error": "Failed to handle user"}), 500
            
        # 데이터 형식을 DB ENUM과 일치하도록 변환
        formatted_data = {
            "user_input": data.get('user_input', ''),
            "tags": data.get('tags', {}),
            "created_title": data.get('created_title', ''),
            "created_content": data.get('created_content', ''),
            "similar_1": data.get('similar_1', {}),
            "similar_2": data.get('similar_2', {}),
            "similar_3": data.get('similar_3', {}),
            "recommended_1": data.get('recommended_1', ''),
            "recommended_2": data.get('recommended_2', ''),
            "recommended_3": data.get('recommended_3', '')
        }
            
        conversation_id = StoryService.save_to_database(user_id, data, formatted_data)
        
        if conversation_id is None:
            return jsonify({"error": "Failed to save to database"}), 500
            
        return jsonify({
            "success": True,
            "conversation_id": conversation_id,
            "user_id": user_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/retrieveChatHistoryList', methods=['POST'])
@log_request_response
def retrieve_chat_history_list():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400
            
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({"error": "사용자 ID가 필요합니다"}), 400
            
        result = HistoryService.get_chat_history_list(user_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/retrieveChatHistoryDetail', methods=['POST'])
@log_request_response
def retrieve_chat_history_detail():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        thread_id = data.get('thread_id')
        user_id = data.get('user_id')
        
        if not thread_id or not user_id:
            return jsonify({"error": "thread_id and user_id are required"}), 400
            
        result = HistoryService.get_chat_history_detail(thread_id, user_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/searchHistory', methods=['POST'])
@log_request_response
def search_history():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400
            
        user_id = data.get('user_id')
        search_text = data.get('search_text', '').strip()
        
        if not user_id:
            return jsonify({"error": "사용자 ID가 필요합니다"}), 400
        if not search_text:
            return jsonify({"error": "검색어가 필요합니다"}), 400
            
        result = HistoryService.search_history(user_id, search_text)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/updateEvaluation', methods=['POST'])
@log_request_response
def update_evaluation():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400
            
        required_fields = ['thread_id', 'conversation_id', 'user_id', 'evaluation']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "필수 필드가 누락되었습니다"}), 400
            
        # evaluation 값 검증
        if data['evaluation'] not in ['like', 'dislike']:
            return jsonify({"error": "잘못된 evaluation 값입니다"}), 400
            
        result = EvaluationService.update_content_evaluation(
            thread_id=data['thread_id'],
            conversation_id=data['conversation_id'],
            user_id=data['user_id'],
            evaluation=data['evaluation']
        )
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/getEvaluation', methods=['POST'])
@log_request_response
def get_evaluation():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400
            
        required_fields = ['thread_id', 'conversation_id', 'user_id']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "필수 필드가 누락되었습니다"}), 400
            
        result = EvaluationService.get_content_evaluation(
            thread_id=data['thread_id'],
            conversation_id=data['conversation_id'],
            user_id=data['user_id']
        )
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/deleteThread', methods=['POST'])
@log_request_response
def delete_thread():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400
            
        thread_id = data.get('thread_id')
        user_id = data.get('user_id')
        
        if not thread_id or not user_id:
            return jsonify({"error": "thread_id와 user_id가 필요합니다"}), 400
            
        result = StoryService.delete_thread(thread_id, user_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/updateThreadTitle', methods=['POST'])
@log_request_response
def update_thread_title():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400
            
        thread_id = data.get('thread_id')
        user_id = data.get('user_id')
        title = data.get('title')
        
        if not all([thread_id, user_id, title]):
            return jsonify({"error": "thread_id, user_id, title이 모두 필요합니다"}), 400
            
        result = HistoryService.update_thread_title(thread_id, user_id, title)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({"error": result['error']}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/cancelGeneration', methods=['POST'])
@log_request_response
def cancel_generation():
    try:
        with Database() as db:
            data = request.json
            if not data:
                return jsonify({"error": "데이터가 제공되지 않았습니다"}), 400
                
            thread_id = data.get('thread_id')
            conversation_id = data.get('conversation_id')
            
            if not all([thread_id, conversation_id]):
                return jsonify({"error": "필수 파라미터가 누락되었습니다"}), 400
                
            result = StoryService.cancel_generation(thread_id, conversation_id)
            
            if result.get("success"):
                try:
                    # user_input과 tags 조회
                    db.execute(
                        """SELECT 
                            MAX(CASE WHEN category = 'user_input' THEN data END) as user_input,
                            MAX(CASE WHEN category = 'tags' THEN data END) as tags
                           FROM conversation_data 
                           WHERE thread_id = %s 
                           AND conversation_id = %s
                           AND category IN ('user_input', 'tags')
                           GROUP BY thread_id, conversation_id""",
                        (thread_id, conversation_id)
                    )
                    db_result = db.fetchone()
                    
                    return jsonify({
                        "success": True,
                        "message": "Generation cancelled successfully",
                        "thread_id": thread_id,
                        "conversation_id": conversation_id,
                        "user_input": db_result['user_input'] if db_result else "",
                        "tags": json.loads(db_result['tags']) if db_result and db_result['tags'] else {}
                    })
                except Exception as db_error:
                    current_app.logger.error(f"Database error while fetching user_input and tags: {str(db_error)}")
                    return jsonify({
                        "success": True,
                        "message": "Generation cancelled successfully but failed to fetch additional data",
                        "thread_id": thread_id,
                        "conversation_id": conversation_id,
                        "user_input": "",
                        "tags": {}
                    })
            else:
                return jsonify({"error": result.get("error", "이야기 생성 중단에 실패했습니다")}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500