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