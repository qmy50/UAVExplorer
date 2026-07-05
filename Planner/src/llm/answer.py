from llm.utils.only_answer import only_answer


def get_answer(client, prompt=None):
    if client.llm_client == 'deepseek':
        try:
            from llm.client.deepseek_answer import deepseek_respond
            respond = deepseek_respond(prompt=prompt)
        except ImportError as e:
            print(f"[LLM] deepseek unavailable: {e}")
            respond = []
    # elif client.llm_client == 'ollama':
    #     try:
    #         from llm.client.ollama_answer import ollama_respond
    #         respond = ollama_respond(model=client.ollama, prompt=prompt)
    #     except ImportError as e:
    #         print(f"[LLM] ollama unavailable: {e}")
    #         respond = []
    else:
        respond = []

    if not respond or not isinstance(respond, str):
        print(f"[LLM] No valid response (type={type(respond).__name__}), returning empty")
        return [], respond

    similar_answer = only_answer(respond)
    if similar_answer is None:
        print(f"[LLM] Failed to parse answer from response: {str(respond)[:200]}...")
        return [], respond
    return similar_answer, respond
