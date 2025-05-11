# 📚 Story Tool Backend

**고전 스토리 생성 툴**의 백엔드는 **Flask 기반의 Python 웹 서버**로 구성되어 있으며,  
**Ubuntu 서버 환경에 배포**되어 운영되고 있습니다.  
사용자 입력을 바탕으로 OpenAI GPT 모델과 자체 생성기를 활용하여  
실시간 스트리밍 방식의 고전 스타일 이야기를 생성하고,  
유사 문단 검색 및 분석 기능까지 제공하는 **AI 기반 스토리 생성 시스템**입니다.

> 🟢 **배포 주소**: [스토리생성툴로 이동하기](http://202.86.11.19:8012/)

---


## 🧾 Overview

이 백엔드는 다음과 같은 핵심 기능을 수행합니다:

- **하이브리드 스토리 생성**: Fine-tuned 모델 + OpenAI Assistant API
- **실시간 스트리밍 응답**: 콘텐츠 청크 단위 전송
- **리트리버 기반 유사 문단 검색**
- **대화 히스토리 관리** (Thread & Conversation)
- **콘텐츠 평가 및 피드백 수집**
- **MySQL + FAISS 기반 벡터 검색 시스템**

---

## 🏗️ 시스템 아키텍처

![Architecture](https://github.com/user-attachments/assets/1b364f2b-1f33-425f-9b1b-3e502ee6296c)
- **Client Layer**: React 기반 프론트엔드에서 HTTP 요청
- **Web Layer**: Flask 서버 및 API 라우팅
- **Service Layer**: StoryService, SearchService, AnalysisService 등 비즈니스 로직
- **Manager Layer**: ThreadManager, ConversationManager, OpenAIAssistantManager
- **Data Layer**: MySQL 기반 DB와 FAISS 벡터 검색

---

## 🔄 생성 취소 흐름 (Cancel Flow)

![Cancel Flow](https://github.com/user-attachments/assets/2dcb0533-36e0-43ed-92fb-c5a48b803eaf)

- `/api/cancelGeneration` 요청 → OpenAI run 취소
- 취소 플래그 설정 및 DB 상태 업데이트
- 사용자 입력 및 태그와 함께 응답 반환

---

## ⚙ 초기화 및 실행 흐름

![Initialization](https://github.com/user-attachments/assets/472d71cf-1dc1-40c6-abae-781196ddea9b)

- `run.py` → `create_app()` → Flask 인스턴스 생성
- `init_app()` 내부에서 OpenAI Assistant 초기화 수행
- `api_bp` 블루프린트 등록 후 서버 실행

---

## 🧩 주요 컴포넌트

### ✅ `StoryService`

- 전체 스토리 생성 흐름의 중심 클래스
- 주요 메서드:
  - `initialize()`
  - `hybrid_generate_story_with_assistant()`
  - `cancel_generation()`
  - `search_and_recommend()`

### ✅ `ThreadManager`

- OpenAI 스레드 ID 생성 및 관리
- 스레드 제목 업데이트, 삭제 등

### ✅ `ConversationManager`

- 대화 ID 생성 및 상태 관리
- 생성 완료 여부, 취소 처리

### ✅ `StoryGenerator`

- Fine-tuned 모델을 통한 기본 스토리 생성
- 추천/제목 생성 로직 포함

### ✅ `OpenAIAssistantManager`

- OpenAI Assistant 생성 및 호출 관리

---

## 📡 API 엔드포인트

| 경로 | 기능 |
|------|------|
| `POST /api/generate` | 스토리 생성 (스트리밍) |
| `POST /api/search` | 유사 문단 검색 및 추천 |
| `POST /api/analyze` | 스토리 내용 분석 |
| `GET /api/retrieveChatHistoryList` | 사용자 히스토리 목록 조회 |
| `GET /api/retrieveChatHistoryDetail` | 세부 대화 내용 조회 |
| `POST /api/cancelGeneration` | 생성 취소 요청 |
| `POST /api/updateEvaluation` | 사용자 피드백 등록 |
| `DELETE /api/deleteThread` | 대화 스레드 삭제 |

---

## 🗃 데이터 모델

- `threads`: 사용자별 대화 주제, OpenAI thread ID 포함
- `conversations`: 대화 세션, 상태값 포함 (`in_progress`, `cancelled`, `completed`)
- `conversation_data`: 입력/생성된 스토리/제목/추천 등 저장
- `content_evaluation`: 좋아요/싫어요 등 사용자 평가 정보 저장

---

## ⚙ 구성 설정 (`app/config.py`)

| 항목 | 설명 |
|------|------|
| 모델 설정 | FINE_TUNED_MODEL, GPT_4O_MODEL 등 |
| 파일 경로 | 토픽 맵, 벡터 저장 위치 |
| DB 연결 | MySQL DB 접속 정보 |
| API 연동 | 외부 API 키 및 URL |
| 시스템 프롬프트 | 이야기 생성 시 기본 템플릿 |

---

## 🤖 OpenAI 통합

- Fine-tuned 모델: 스토리의 기본 틀 생성
- GPT-4o: Assistant API 기반 확장/정제
- GPT-mini: 추천 및 제목 생성

> **Streaming API** 사용으로 실시간 사용자 응답 제공

---

## ✅ 예시 흐름 (Generate)

```plaintext
Client → /api/generate
 → Thread/Conversation 생성
 → Base story 생성
 → GPT Assistant Run (Streaming)
 → 청크 수신 및 전송
 → 제목/추천 생성
 → 저장 및 완료 응답
```

---

## 📎 관련 소스

| 파일 | 설명 |
|------|------|
| `app/services/story_service.py` | 스토리 생성 메인 로직 |
| `app/services/thread_manager.py` | 스레드 관리 |
| `app/services/conversation_manager.py` | 대화 상태 관리 |
| `app/services/story_generator.py` | 스토리 생성기 |
| `app/services/openai_assistant_manager.py` | OpenAI 연동 |
| `app/api/routes.py` | API 라우팅 정의 |
| `app/config.py` | 전체 시스템 설정 관리 |

---

## 🏁 마무리

본 시스템은 구조화된 레이어를 기반으로 고전 문학 스타일의 이야기 생성, 유사 문단 검색, 평가 기능을 모두 통합한 백엔드 시스템입니다.  
OpenAI API를 적극 활용하면서도, 자체 DB 및 벡터 스토어를 통해 유연한 기능 확장이 가능합니다.

---

