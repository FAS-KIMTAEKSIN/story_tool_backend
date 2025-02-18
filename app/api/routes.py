from flask import Blueprint, request, jsonify, Response, stream_with_context
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

api_bp = Blueprint('api', __name__)

@api_bp.route('/search', methods=['POST'])
@log_request_response
def search_documents():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        results = SearchService.search_documents(data)
        
        # 응답 형식을 similar_1, 2, 3 키를 가진 객체로 변환
        return jsonify({
            "success": True,
            "result": {
                "similar_1": results[0] if len(results) > 0 else "",
                "similar_2": results[1] if len(results) > 1 else "",
                "similar_3": results[2] if len(results) > 2 else ""
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
            return jsonify({"error": "No data provided"}), 400
            
        # 스토리 생성
        generated_result = StoryService.generate_story(data)
        
        # 응답 형식을 DB ENUM과 일치하도록 변환
        result = {
            "user_input": data.get('user_input', ''),
            "tags": data.get('tags', {}),
            "created_title": generated_result.get('created_title', ''),
            "created_content": generated_result.get('created_content', ''),
            "similar_1": "",  # 프론트에서 /search 결과로 채워질 예정
            "similar_2": "",  # 프론트에서 /search 결과로 채워질 예정
            "similar_3": "",  # 프론트에서 /search 결과로 채워질 예정
            "recommended_1": generated_result.get('recommendations', [''])[0],
            "recommended_2": generated_result.get('recommendations', ['', ''])[1],
            "recommended_3": generated_result.get('recommendations', ['', '', ''])[2]
        }
        
        return jsonify({
            "success": True,
            "result": result
        })
        
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

@api_bp.route('/generateWithSearch', methods=['POST'])
@log_request_response
def generate_with_search():
    try:
        data = request.json
        print(f"[DEBUG] Received request data: {data}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        def generate():
            try:
                # 스트림 시작을 알림
                yield "data: {\"status\": \"generating\"}\n\n"
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # 검색 시작
                    search_future = executor.submit(SearchService.search_documents, data)
                    
                    # 이야기 생성 및 실시간 스트리밍
                    story_generator = StoryService.generate_story(data)
                    final_content = None
                    
                    # 생성되는 내용을 실시간으로 전송
                    for content in story_generator:
                        if isinstance(content, dict):  # 최종 결과인 경우
                            generated_result = content
                            final_content = content['created_content']
                            break
                        else:  # 생성 중인 컨텐츠인 경우
                            final_content = content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    
                    # 검색 결과 대기
                    search_results = search_future.result()
                
                # 최종 결과 병합
                result = {
                    "user_input": data.get('user_input', ''),
                    "tags": data.get('tags', {}),
                    "created_title": generated_result['created_title'],
                    "created_content": final_content,
                    "similar_1": search_results[0] if len(search_results) > 0 else {},
                    "similar_2": search_results[1] if len(search_results) > 1 else {},
                    "similar_3": search_results[2] if len(search_results) > 2 else {},
                    "recommended_1": generated_result['recommendations'][0],
                    "recommended_2": generated_result['recommendations'][1],
                    "recommended_3": generated_result['recommendations'][2]
                }
                
                # 임시 사용자 ID 가져오기
                user_id = Database.get_or_create_temp_user()
                if user_id is None:
                    raise Exception("Failed to handle user")
                    
                # DB에 저장
                thread_id, conversation_id = StoryService.save_to_database(user_id, data, result)
                if thread_id is None or conversation_id is None:
                    raise Exception("Failed to save to database")
                
                # 최종 결과 전송
                final_result = {
                    "success": True,
                    "result": result,
                    "thread_id": thread_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id
                }
                yield f"data: {json.dumps(final_result)}\n\n"
                
            except Exception as e:
                print(f"[DEBUG] Error in generate function: {str(e)}")
                print(f"[DEBUG] Error traceback: {traceback.format_exc()}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
            
        result = HistoryService.delete_thread(thread_id, user_id)
        
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