import pytest

from src.services.ai_response_parser import (
    extract_ai_response_content,
    parse_ai_response_json,
    EmptyAIResponseError,
)


def test_parse_ai_response_json_uses_first_object_when_multiple_json_objects_are_concatenated():
    content = """```json
{"is_recommended": true, "reason": "first"}
{"is_recommended": false, "reason": "second"}
```"""

    result = parse_ai_response_json(content)

    assert result == {"is_recommended": True, "reason": "first"}


def test_parse_ai_response_json_extracts_json_from_wrapped_text():
    content = """分析结果如下：

```json
{"is_recommended": true, "reason": "wrapped"}
```

请按第一份结果处理。"""

    result = parse_ai_response_json(content)

    assert result == {"is_recommended": True, "reason": "wrapped"}


def test_parse_ai_response_json_raises_when_no_json_exists():
    with pytest.raises(ValueError):
        parse_ai_response_json("没有任何 JSON 内容")


def test_extract_ai_response_content_with_none_content_but_valid_reasoning_content():
    """当 content 为 None 但 reasoning_content 有值时，应该成功提取 reasoning_content 的内容"""
    # 创建 mock 对象模拟 OpenAI 风格的响应
    message = type('Message', (), {
        'content': None,
        'reasoning_content': '这是推理内容'
    })()
    choice = type('Choice', (), {'message': message})()
    response = type('Response', (), {'choices': [choice]})()

    result = extract_ai_response_content(response)

    assert result == '这是推理内容'


def test_extract_ai_response_content_reassembles_sse_stream_body():
    """部分转发网关对非流式请求也返回 SSE 流，原始流文本应被重组为完整内容。"""
    sse_body = (
        'data: {"id":"1","object":"chat.completion.chunk","choices":'
        '[{"index":0,"delta":{"content":"### 第一"},"finish_reason":null}]}\n\n'
        'data: {"id":"1","object":"chat.completion.chunk","choices":'
        '[{"index":0,"delta":{"content":"部分"},"finish_reason":null}]}\n\n'
        'data: {"id":"1","object":"chat.completion.chunk","choices":'
        '[{"index":0,"delta":{"content":"：核心原则"},"finish_reason":null}]}\n\n'
        'data: [DONE]\n\n'
    )

    result = extract_ai_response_content(sse_body)

    assert result == "### 第一部分：核心原则"


def test_extract_ai_response_content_reassembles_sse_stream_with_reasoning_content():
    """SSE 流仅在 reasoning_content 中携带内容时也应被正确重组。"""
    sse_body = (
        'data: {"choices":[{"delta":{"reasoning_content":"思考A"}}]}\n\n'
        'data: {"choices":[{"delta":{"reasoning_content":"思考B"}}]}\n\n'
        'data: [DONE]\n\n'
    )

    result = extract_ai_response_content(sse_body)

    assert result == "思考A思考B"


def test_extract_ai_response_content_with_plain_string_is_unchanged():
    """普通字符串（非 SSE）应原样返回，不受重组逻辑影响。"""
    result = extract_ai_response_content("这是一段普通的分析标准文本")

    assert result == "这是一段普通的分析标准文本"


def test_extract_ai_response_content_raises_when_content_and_reasoning_content_are_empty():
    """当 content 和 reasoning_content 都为空时，应该抛出 EmptyAIResponseError"""
    # 创建 mock 对象
    message = type('Message', (), {
        'content': None,
        'reasoning_content': None
    })()
    choice = type('Choice', (), {'message': message})()
    response = type('Response', (), {'choices': [choice]})()

    with pytest.raises(EmptyAIResponseError):
        extract_ai_response_content(response)
