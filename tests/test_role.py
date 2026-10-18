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

    # Input
    data = {'text': '请查询任务DF-0012的当前状态，列出可用的只读工具。'}

    # Execute
    result = await execute(data, ctx)

    # Assert
    assert result['decision'] == 'PASS'
    assert 'answer' in result
    assert '任务 DF-0012 当前状态：已完成。可用只读工具：log-viewer、trace-explorer。' in result['answer']
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
