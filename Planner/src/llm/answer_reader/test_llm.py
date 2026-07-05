#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/tianbot/explorer_ws/src/Planner/src')
from llm.answer_reader.answer_reader import read_answer

class C:
    llm_client = 'deepseek'
    ollama = 'qwen2.5'

# 测试缓存里已有的标签
ans, room, thresh = read_answer(
    
'/home/tianbot/explorer_ws/src/Planner/src/llm/answers/llm_answer_hm3d.txt',
    '/tmp/llm_test_responses.txt',
    'toilet',  # 'bed'
    C()
)
print(f'易混淆标签: {ans}')
print(f'房间:        {room}')
print(f'融合阈值:    {thresh}')
