import random
import string

def parse_content_to_json(content):
    """주어진 content 문자열을 JSON 형식으로 변환"""
    if not content:
        return {}
        
    lines = content.strip().split("\n")
    parsed_data = {}
    for line in lines:
        if " : " in line:
            key, value = line.split(" : ", 1)
            parsed_data[key.strip()] = value.strip()
    return parsed_data 

def generate_session_hash(length=10):
    """무작위 세션 해시를 생성합니다."""
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))