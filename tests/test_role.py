import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List


# Import target function
from role import execute, _blocked


@pytest.mark.asyncio
async def test_execute_normal_case():
    # Mock context
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    # Simulate prompt compile
    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = "You are a task navigator..."
    ctx.prompt.return_value = mock_prompt

    # Tool returns
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0012', 'status': 'DONE', 'title': 'Fix login bug'}],
        [
            {'name': 'log-viewer', 'access': 'READ', 'risk': 'LOW'},
            {'name': 'trace-explorer', 'access': 'READ', 'risk': 'LOW'},
            {'name': 'config-dump', 'access': 'WRITE', 'risk': 'MID'}
        ]
    ]

    # LLM returns PASS
    ctx.llm.chat.return_value = '{"decision": "PASS"}'

    # Input: covers Chinese-adjacent DF- (h-3)
    data = {'text': '任务DF-0012现在怎样？给我当前记录与目录建议。'}

    # Execute
    result = await execute(data, ctx)

    # Assert
    assert result['decision'] == 'PASS'
    assert 'answer' in result
    assert '已完成' in result['answer']
    assert 'log-viewer' in result['answer']
    assert len(result['evidence_sources']) == 2
    assert len(result['steps']) == 4


@pytest.mark.asyncio
async def test_execute_blocked_no_safe_tools():
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0012', 'status': 'DONE'}],
        [
            {'name': 'config-dump', 'access': 'WRITE', 'risk': 'MID'},
            {'name': 'exec-runner', 'access': 'EXEC', 'risk': 'HIGH'}
        ]
    ]

    data = {'text': 'DF-0012'}

    result = await execute(data, ctx)

    assert result['decision'] == 'BLOCKED'
    assert '无符合 READ/LOW 要求的可用工具' in result['block_reason']
    assert len(result['evidence_sources']) == 2
    assert len(result['steps']) == 4


@pytest.mark.asyncio
async def test_execute_blocked_no_df():
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."

    data = {'text': '请查询任务当前进度并给出工具建议。'}

    result = await execute(data, ctx)

    assert result['decision'] == 'BLOCKED'
    assert '必须且仅能指定一个DF编号' in result['block_reason']
    assert len(result['evidence_sources']) == 0
    assert len(result['steps']) == 0


@pytest.mark.asyncio
async def test_execute_blocked_multiple_df():
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."

    data = {'text': '查询任务DF-0012和DF-0013的状态'}

    result = await execute(data, ctx)

    assert result['decision'] == 'BLOCKED'
    assert '匹配到2个' in result['block_reason']
    assert len(result['evidence_sources']) == 0
    assert len(result['steps']) == 0


@pytest.mark.asyncio
async def test_execute_blocked_job_not_found():
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [[], [{'name': 't1', 'access': 'READ', 'risk': 'LOW'}]]

    data = {'text': '查询DF-999999'}

    result = await execute(data, ctx)

    assert result['decision'] == 'BLOCKED'
    assert '未查到任务' in result['block_reason']
    assert len(result['evidence_sources']) == 0
    assert len(result['steps']) == 0


@pytest.mark.asyncio
async def test_execute_blocked_empty_catalog():
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [[{'status': 'DONE'}], []]

    data = {'text': 'DF-0001'}

    result = await execute(data, ctx)

    assert result['decision'] == 'BLOCKED'
    assert '工具目录为空' in result['block_reason']
    assert len(result['evidence_sources']) == 0
    assert len(result['steps']) == 0


@pytest.mark.asyncio
async def test_execute_llm_returns_blocked_stays_blocked():
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0012', 'status': 'FAIL'}],
        [{'name': 'log-viewer', 'access': 'READ', 'risk': 'LOW'}]
    ]
    ctx.llm.chat.return_value = '{"decision": "BLOCKED", "block_reason": "PASS is not allowed here"}'

    data = {'text': 'DF-0012'}

    result = await execute(data, ctx)

    assert result['decision'] == 'BLOCKED'
    assert result['block_reason'] == 'PASS is not allowed here'
    assert len(result['evidence_sources']) == 2
    assert len(result['steps']) == 4


@pytest.mark.asyncio
async def test_execute_blocked_empty_safe_tools_from_manifest_compliance():
    # Regression: ensure safe_tools filtering respects trusted agent.yaml enums, not tool result dict keys
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    # Simulate tool record with lowercase/whitespace-padded enums — must normalize per contract
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0012', 'status': 'DONE'}],
        [
            {'name': 'log-viewer', 'access': ' read ', 'risk': 'low'},
            {'name': 'config-dump', 'access': 'WRITE', 'risk': 'MID'}
        ]
    ]
    ctx.llm.chat.return_value = '{"decision": "PASS"}'

    data = {'text': 'DF-0012'}

    result = await execute(data, ctx)

    assert result['decision'] == 'PASS'
    assert 'answer' in result
    assert '已完成' in result['answer']
    assert 'log-viewer' in result['answer']
    assert len(result['evidence_sources']) == 2
    assert len(result['steps']) == 4


@pytest.mark.asyncio
async def test_execute_blocked_job_record_missing_status():
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0012'}],  # missing 'status'
        [{'name': 'log-viewer', 'access': 'READ', 'risk': 'LOW'}]
    ]

    data = {'text': 'DF-0012'}

    result = await execute(data, ctx)

    assert result['decision'] == 'BLOCKED'
    assert '缺少 status 字段' in result['block_reason']
    assert len(result['evidence_sources']) == 2
    assert len(result['steps']) == 4


@pytest.mark.asyncio
async def test_execute_normal_case_with_negation_safety():
    # h-4: "查询任务DF-0011，仅解释现状和工具，不执行发布。"
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0011', 'status': 'WAIT'}],
        [{'name': 'log-viewer', 'access': 'READ', 'risk': 'LOW'}]
    ]
    ctx.llm.chat.return_value = '{"decision": "PASS"}'

    data = {'text': '查询任务DF-0011，仅解释现状和工具，不执行发布。'}

    result = await execute(data, ctx)

    assert result['decision'] == 'PASS'
    assert 'answer' in result
    assert '等待中' in result['answer']
    assert len(result['evidence_sources']) == 2
    assert len(result['steps']) == 4


@pytest.mark.asyncio
async def test_execute_blocked_invalid_llm_json():
    # Test invalid JSON: null, list, bool, string, object without decision
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0012', 'status': 'DONE'}],
        [{'name': 'log-viewer', 'access': 'READ', 'risk': 'LOW'}]
    ]

    # Test cases: each should BLOCKED with evidence_sources/steps non-empty
    invalid_cases = [
        'null',
        '[1,2,3]',
        'true',
        '"hello"',
        '{}',
        '{"answer":"x"}',  # no decision
        '{"decision": 123}',  # decision not string
        '{"decision": "INVALID"}',
        '{"decision": "PASS", "answer": 123}'  # answer not string
    ]

    for invalid_json in invalid_cases:
        ctx.llm.chat.return_value = invalid_json
        data = {'text': 'DF-0012'}
        result = await execute(data, ctx)
        assert result['decision'] == 'BLOCKED'
        assert len(result['evidence_sources']) == 2
        assert len(result['steps']) == 4


@pytest.mark.asyncio
async def test_execute_chinese_punctuation_adjacent():
    # Test Chinese punctuation adjacency: h-15, h-17, h-18 cover edge cases; this verifies extraction works
    ctx = MagicMock()
    ctx.prompt = AsyncMock()
    ctx.tools.call = AsyncMock()
    ctx.llm.chat = AsyncMock()

    ctx.prompt.return_value = MagicMock()
    ctx.prompt.return_value.compile.return_value = "..."
    ctx.tools.call.side_effect = [
        [{'jobId': 'DF-0012', 'status': 'RUN'}],
        [{'name': 'log-viewer', 'access': 'READ', 'risk': 'LOW'}]
    ]
    ctx.llm.chat.return_value = '{"decision": "PASS"}'

    # Test cases from H1/H2: all must extract exactly one DF-
    test_cases = [
        '任务DF-0012。',
        'DF-0012状态？',
        '请看DF-0012！',
        'DF-0012，谢谢',
        '任务编号是DF-0012，确认一下',
    ]

    for text in test_cases:
        data = {'text': text}
        result = await execute(data, ctx)
        assert result['decision'] == 'PASS'
        assert len(result['evidence_sources']) == 2
        assert len(result['steps']) == 4