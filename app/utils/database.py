import mysql.connector
from mysql.connector import Error
from app.config import Config

class Database:
    @staticmethod
    def get_connection():
        try:
            connection = mysql.connector.connect(
                host=Config.DB_HOST,
                user=Config.DB_USER,
                password=Config.DB_PASSWORD,
                database=Config.DB_NAME,
                port=3306,  # MySQL 기본 포트
                connect_timeout=10,  # 연결 타임아웃 10초
                time_zone='+09:00'  # KST (UTC+9) 설정
            )
            
            # 세션 타임존 설정
            cursor = connection.cursor()
            cursor.execute("SET time_zone='+09:00'")
            cursor.close()
            
            return connection
        except Error as e:
            print(f"Error connecting to MySQL Database: {str(e)}")
            return None

    @staticmethod
    def get_or_create_temp_user():
        """임시 사용자 조회 또는 생성"""
        connection = Database.get_connection()
        if not connection:
            return None

        try:
            cursor = connection.cursor(dictionary=True)
            
            # 임시 사용자 조회
            cursor.execute(
                "SELECT user_id FROM users WHERE user_name = 'temp_user' LIMIT 1"
            )
            user = cursor.fetchone()
            
            if user:
                return user['user_id']
                
            # 새로운 user_id 생성
            cursor.execute("SELECT MAX(user_id) FROM users")
            max_user_id = cursor.fetchone()['MAX(user_id)']
            new_user_id = (max_user_id or 0) + 1
            
            # 임시 사용자 생성
            cursor.execute(
                """INSERT INTO users 
                   (user_id, user_name, password) 
                   VALUES (%s, %s, %s)""",
                (new_user_id, 'temp_user', 'temp_password')  # 실제 구현시 해시된 비밀번호 사용
            )
            connection.commit()
            return new_user_id
            
        except Exception as e:
            print(f"Error handling temp user: {e}")
            if connection:
                connection.rollback()
            return None
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    def __init__(self):
        self.connection = None
        self.cursor = None

    def __enter__(self):
        self.connection = self.get_connection()
        if self.connection:
            self.cursor = self.connection.cursor(dictionary=True)
            return self
        raise Exception("Database connection failed")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.connection:
                self.connection.rollback()
        else:
            if self.connection:
                self.connection.commit()
        
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    @staticmethod
    def execute_transaction(callback):
        """트랜잭션으로 실행할 콜백 함수를 처리합니다"""
        connection = Database.get_connection()
        if not connection:
            return None
            
        try:
            cursor = connection.cursor(dictionary=True)
            # 트랜잭션 시작
            connection.start_transaction(isolation_level='REPEATABLE READ')
            
            # 콜백 실행
            result = callback(cursor)
            
            # 트랜잭션 커밋
            connection.commit()
            return result
            
        except Exception as e:
            print(f"Transaction error: {str(e)}")
            if connection:
                connection.rollback()
            raise
            
        finally:
            if connection and connection.is_connected():
                cursor.close()
                connection.close()

    @staticmethod
    def get_next_thread_id(cursor, user_id):
        """다음 thread_id를 안전하게 가져옵니다"""
        cursor.execute(
            """SELECT MAX(thread_id) + 1 as next_id 
               FROM threads 
               FOR UPDATE"""
        )
        next_id = cursor.fetchone()['next_id'] or 1
        return next_id

    @staticmethod
    def get_next_conversation_id(cursor, thread_id):
        """다음 conversation_id를 안전하게 가져옵니다"""
        cursor.execute(
            """SELECT MAX(conversation_id) + 1 as next_id 
               FROM conversations 
               WHERE thread_id = %s 
               FOR UPDATE""",
            (thread_id,)
        )
        next_id = cursor.fetchone()['next_id'] or 1
        return next_id

    def execute(self, query, params=None):
        try:
            self.cursor.execute(query, params)
            return self.cursor
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            raise

    @property
    def lastrowid(self):
        return self.cursor.lastrowid

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall() 