const generateStory = async () => {
  setIsLoading(true);
  setGeneratedContent(''); // 컨텐츠 초기화
  
  try {
    const response = await fetch('/api/generateWithSearch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Reader not available');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // 청크 디코딩
      const chunk = new TextDecoder().decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          
          if (data.status === 'generating') {
            // 생성 시작
            setIsLoading(true);
          } 
          else if (data.content) {
            // 이야기 내용 업데이트
            setGeneratedContent(data.content);
          }
          else if (data.success) {
            // 최종 결과 처리
            setIsLoading(false);
            setResult(data.result);
            setThreadId(data.thread_id);
            setConversationId(data.conversation_id);
            setUserId(data.user_id);
          }
          else if (data.error) {
            throw new Error(data.error);
          }
        }
      }
    }
  } catch (error) {
    console.error('Error:', error);
    setError(error.message);
    setIsLoading(false);
  }
}; 