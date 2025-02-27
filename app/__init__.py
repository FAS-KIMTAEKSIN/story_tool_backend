# 빈 파일로 두어도 됩니다 

from app.services.story_service import StoryService

# 다른 초기화 코드...

# Assistant 초기화 - 애플리케이션 시작 시 한 번만 실행
def init_app():
    StoryService.initialize_assistants()

# 초기화 함수 호출
init_app() 