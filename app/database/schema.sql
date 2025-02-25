-- 사용자 테이블
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,             -- 내부 관리용 고유 ID
    user_id INT NOT NULL,                          -- 사용자 ID (로그인용)
    user_name VARCHAR(255) NOT NULL,               -- 사용자 이름 또는 이메일
    password VARCHAR(255) NOT NULL,                -- 비밀번호 (해시된 값)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 계정 생성 시간
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_user_id (user_id)            -- user_id 유일성 보장
);
-- 대화창 테이블
CREATE TABLE threads (
    thread_id INT AUTO_INCREMENT PRIMARY KEY,      -- 대화창 고유 ID
    user_id INT NOT NULL,                          -- 사용자 ID
    title VARCHAR(255),                            -- 대화창 제목
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 대화 생성 시간
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
-- 대화 테이블
CREATE TABLE conversations (
    id INT AUTO_INCREMENT PRIMARY KEY,             -- 내부 관리용 고유 ID
    conversation_id INT NOT NULL,                  -- 스레드 내 대화 순서
    thread_id INT NOT NULL,                        -- 대화창 ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_thread_conversation (thread_id, conversation_id),
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
);
-- 대화 내용 저장 테이블
CREATE TABLE conversation_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT NOT NULL,                  -- 대화 순서
    thread_id INT NOT NULL,                        -- 대화창 ID
    category ENUM(
        'user_input',                             -- 사용자 입력 내용
        'tags',                                   -- 내용분류
        'created_title',                          -- 생성된 단락의 제목
        'created_content',                        -- 생성된 단락
        'similar_1',                              -- 유사 단락1
        'similar_2',                              -- 유사 단락2
        'similar_3',                              -- 유사 단락3
        'recommended_1',                          -- 이야기 추천1
        'recommended_2',                          -- 이야기 추천2
        'recommended_3'                           -- 이야기 추천3
    ) NOT NULL,
    data TEXT NOT NULL,                           -- 각 카테고리별 저장될 데이터
    FOREIGN KEY (thread_id, conversation_id) REFERENCES conversations(thread_id, conversation_id) ON DELETE CASCADE
);
-- 콘텐츠 평가 테이블
CREATE TABLE content_evaluation (
    thread_id INT NOT NULL,                          -- 대화창 ID
    conversation_id INT NOT NULL,                    -- 대화 순서
    user_id INT NOT NULL,                           -- 평가를 한 사용자 ID
    evaluation ENUM('like', 'dislike') NOT NULL,    -- 평가 (좋아요/싫어요)
    PRIMARY KEY (thread_id, conversation_id),
    FOREIGN KEY (thread_id, conversation_id) 
        REFERENCES conversations(thread_id, conversation_id) 
        ON DELETE CASCADE,
    FOREIGN KEY (user_id) 
        REFERENCES users(user_id) 
        ON DELETE CASCADE,
    -- 한 사용자가 하나의 conversation에 대해 하나의 평가만 가능
    UNIQUE KEY unique_user_evaluation (user_id, thread_id, conversation_id)
); 