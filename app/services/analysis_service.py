from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait
from app.config import Config

class AnalysisService:
    client = OpenAI()
    
    @classmethod
    def analyze_section(cls, title, section):
        """작품 분석"""
        completion = cls.client.chat.completions.create(
            model=Config.GPT_MODEL,
            messages=[
                {"role": "system", "content": "당신은 고전 문학 전문가입니다. 주어진 작품에 대해 전문적이고 학술적인 분석을 제공해주세요."},
                {"role": "user", "content": f"{title}의 {section}을 설명해주세요."}
            ]
        )
        return completion.choices[0].message.content

    @classmethod
    def analyze_work(cls, data):
        """작품 분석 통합"""
        title = data.get('title', '')

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(cls.analyze_section, title, section)
                for section in ["개요", "주요 특징", "문학사적 의의"]
            ]
            done, _ = wait(futures)
            results = [f.result() for f in done]

        return {
            "overview": results[0],
            "characteristics": results[1],
            "significance": results[2]
        } 