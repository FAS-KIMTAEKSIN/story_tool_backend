from functools import wraps
from flask import request, jsonify
import logging
from datetime import datetime
import json

def log_request_response(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 요청 로깅
        request_data = request.get_json() if request.is_json else None
        logging.info(f"[{timestamp}] REQUEST to {request.path}:")
        logging.info(f"Request Data: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
        
        # 함수 실행
        response = func(*args, **kwargs)
        
        # 응답이 튜플인 경우 (에러 응답) 처리
        if isinstance(response, tuple):
            response_data = response[0]
            status_code = response[1]
        else:
            response_data = response
            status_code = 200
            
        # 응답 로깅
        logging.info(f"[{timestamp}] RESPONSE from {request.path}:")
        if isinstance(response_data, dict):
            logging.info(f"Response Data: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
        else:
            logging.info(f"Response Data: {response_data}")
        logging.info(f"Status Code: {status_code}")
        
        return response
        
    return wrapper 