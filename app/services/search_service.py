from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
import numpy as np
import json
from app.utils.helpers import parse_content_to_json
from app.config import Config
from app.services.story_service import StoryService
import os
import traceback
from app.utils.database import Database

class SearchService:
    client = OpenAI()
    embedding_model_main = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    embedding_model_topic = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
    
    # 벡터 저장소 로드
    vectorstore_main = FAISS.load_local(
        os.path.join(Config.VECTOR_DB_DIR, "faiss_index_main_blank_keyword"),
        embeddings=embedding_model_main,
        allow_dangerous_deserialization=True
    )
    vectorstore_topic = FAISS.load_local(
        os.path.join(Config.VECTOR_DB_DIR, "faiss_index_topic_blank_keyword"),
        embeddings=embedding_model_topic,
        allow_dangerous_deserialization=True
    )
    
    # 매핑 사전 로드
    with open(Config.TOPIC_MAPPING_PATH, "r", encoding="utf-8") as f:
        topic_mapping = json.load(f)

    @classmethod
    def process_input(cls, data):
        """입력 데이터 처리"""
        theme = data.get('user_input', '').strip()
        if not theme:
            raise ValueError("주제문이 비어있습니다.")
            
        tags = data.get('tags', {})
        
        # 테마에서 키워드 추출
        keywords = cls.extract_keywords_from_theme(theme)
        print(f"Extracted keywords: {keywords}")  # 디버깅용

        # 선택된 태그가 있는 경우에만 분류 정보 추가
        classifications = []
        if tags:
            for category, values in tags.items():
                if values:
                    category_kr = Config.KEY_MAPPING.get(category, category)
                    values_str = ', '.join(values) if isinstance(values, list) else str(values)
                    classifications.append(f"{category_kr}: {values_str}")

        # 분류 정보가 있는 경우와 없는 경우를 구분하여 처리
        if classifications:
            return f"""내용분류: {", ".join(classifications)}
주제어: {keywords}
주제문: {theme}"""
        else:
            return f"""주제어: {keywords}
주제문: {theme}"""

    @classmethod
    def extract_keywords_from_theme(cls, theme):
        """주제문에서 주제어 추출"""
        messages = [
            {
                "role": "system",
                "content": "당신은 주제문에서 핵심 주제어를 추출하는 전문가입니다. "
                          "주제문을 분석하여 3-10개의 핵심 주제어를 추출해주세요. "
                          "주제어는 쉼표로 구분된 단어나 구문으로 반환해주세요. "
                          "다른 설명이나 부가적인 내용 없이 주제어만 반환해주세요."
            },
            {
                "role": "user",
                "content": theme
            }
        ]

        try:
            response = cls.client.chat.completions.create(
                model=Config.GPT_MINI_MODEL,  # GPT_MODEL 대신 GPT_MINI_MODEL 사용
                messages=messages,
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating keywords: {str(e)}")
            return ""

    @classmethod
    def search_documents(cls, data, weight_topic=0.5, k=3):
        """문서 검색 및 JSON 반환"""
        try:
            query_string = cls.process_input(data)
            print(f"Query string: {query_string}")  # 디버깅용

            # 주제어와 주제문 분리
            parts = query_string.split("주제어:")
            if len(parts) > 1:
                main_content = parts[0].strip()  # 내용분류가 있다면 포함
                topic_parts = parts[1].split("주제문:")
                topic_content = topic_parts[0].strip()
                theme_content = topic_parts[1].strip() if len(topic_parts) > 1 else ""
            else:
                main_content = query_string
                topic_content = ""
                theme_content = ""

            print("Separated contents:")  # 디버깅용
            print("Main content:", main_content)
            print("Topic content:", topic_content)
            print("Theme content:", theme_content)

            # 검색 질문을 두 개의 임베딩 모델로 벡터화
            search_text = f"{main_content} {theme_content}".strip()
            query_main_vector = cls.embedding_model_main.embed_query(search_text)
            query_topic_vector = cls.embedding_model_topic.embed_query(topic_content if topic_content else search_text)

            print("Attempting main vector search...")  # 디버깅용
            results_main = cls.vectorstore_main.similarity_search_with_score_by_vector(
                query_main_vector, k=k
            )
            print(f"Main search results count: {len(results_main)}")
            print(f"Main search scores: {[float(score) for _, score in results_main]}")

            print("Attempting topic vector search...")  # 디버깅용
            results_topic = cls.vectorstore_topic.similarity_search_with_score_by_vector(
                query_topic_vector, k=k
            )
            print(f"Topic search results count: {len(results_topic)}")
            print(f"Topic search scores: {[float(score) for _, score in results_topic]}")

            # 결과 처리 및 점수 계산
            scores = {}
            metadata_map = {}

            for doc, score in results_main:
                row_id = doc.metadata.get("row_id")
                if row_id is not None:
                    normalized_score = float(1 / (1 + score))
                    scores[row_id] = scores.get(row_id, 0) + (1 - weight_topic) * normalized_score
                    metadata_map[row_id] = doc.metadata

            for doc, score in results_topic:
                row_id = doc.metadata.get("row_id")
                if row_id is not None:
                    normalized_score = float(1 / (1 + score))
                    scores[row_id] = scores.get(row_id, 0) + weight_topic * normalized_score
                    metadata_map[row_id] = doc.metadata

            # 결과 정렬 및 반환
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            print(f"Combined and normalized scores: {sorted_scores}")  # 디버깅용
            
            if not sorted_scores:
                print("No results found")  # 디버깅용
                return []

            result = []
            for idx, (row_id, score) in enumerate(sorted_scores[:k], 1):
                metadata = metadata_map.get(row_id, {})
                topic_sentence = cls.topic_mapping.get(str(row_id), "")

                content_json = parse_content_to_json("\n".join([f"{k} : {v}" for k, v in metadata.items()]))
                content_json["주제문"] = topic_sentence

                result.append({
                    "document_id": idx,
                    "metadata": metadata,
                    "content": content_json,
                    "score": float(score)
                })

            return result

        except Exception as e:
            print(f"Error in search_documents: {str(e)}")
            raise 

    @classmethod
    def get_recommendations(cls, data):
        """추천 문서 생성"""
        recommendations = [
            StoryService.generate_recommendation(data.get('user_input', ''))
            for _ in range(3)
        ]
        return recommendations

    @classmethod
    def search_and_recommend(cls, data):
        try:
            # 검색 수행
            search_results = cls.search_documents(data)
            
            # 추천은 이미 /generate에서 생성되어 DB에 저장되어 있으므로
            # 여기서는 DB에서 가져오기만 하고 새로 생성하지 않음
            recommendations = []
            try:
                thread_id = data.get('thread_id')
                conversation_id = data.get('conversation_id')
                
                if thread_id and conversation_id:
                    with Database() as cursor:
                        cursor.execute(
                            """SELECT category, data FROM conversation_data 
                               WHERE thread_id = %s AND conversation_id = %s 
                               AND category IN ('recommended_1', 'recommended_2', 'recommended_3')
                               ORDER BY category""",
                            (thread_id, conversation_id)
                        )
                        
                        for row in cursor.fetchall():
                            recommendations.append(row['data'])
            except Exception as e:
                print(f"[ERROR] Failed to retrieve recommendations: {str(e)}")
            
            return {
                'search_results': search_results[:3] if search_results else [],
                'recommendations': recommendations[:3] if recommendations else ["", "", ""]
            }
            
        except Exception as e:
            print(f"[ERROR] Failed in search_and_recommend: {str(e)}")
            print(f"[ERROR] {traceback.format_exc()}")
            return {
                'search_results': [],
                'recommendations': ["", "", ""]
            } 