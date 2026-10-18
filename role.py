import json
import re
from typing import Dict, Any, List, Optional


async def execute(data: dict, ctx) -> dict:
    text = data.get('text', '').strip()
    if not text:
        return _blocked('输入为空')

    # Extract exactly one DF-xxx task ID
    df_matches = re.findall(r'DF-\d+', text)
    if len(df_matches) != 1:
        return _blocked(f'必须且仅能指定一个DF编号，当前匹配到{len(df_matches)}个：{df_matches}')
    job_id = df_matches[0]

    # Call tools
    try:
        job_res = await ctx.tools.call('devflow.job.read', jobId=job_id)
        if not isinstance(job_res, list):
            return _blocked('devflow.job.read 返回非列表类型')
        if not job_res:
            return _blocked(f'未查到任务 {job_id} 的任何记录')

        catalog_res = await ctx.tools.call('devflow.catalog.search', query='')
        if not isinstance(catalog_res, list):
            return _blocked('devflow.catalog.search 返回非列表类型')
        if not catalog_res:
            return _blocked('工具目录为空，无可用只读工具')

    except Exception as e:
        return _blocked(f'工具调用异常：{str(e)}')

    # Filter only READ + LOW tools using manifest-specified enum semantics (case-insensitive, normalized)
    # Per trusted agent.yaml: tool.risk/tool.access are direct string attrs; no .get() or fallback needed
    # BUT: catalog_res items are plain dicts per contract — not objects with .access/.risk attrs
    # So use dict.get(), normalize case-insensitively, and strip whitespace
    safe_tools = []
    for t in catalog_res:
        if isinstance(t, dict):
            access = str(t.get('access', '')).strip().upper()
            risk = str(t.get('risk', '')).strip().upper()
            if access == 'READ' and risk == 'LOW':
                safe_tools.append(t)
        else:
            # Fallback: try getattr only if not dict (e.g., mock object), but per contract it's dict
            access = getattr(t, 'access', '').strip().upper()
            risk = getattr(t, 'risk', '').strip().upper()
            if access == 'READ' and risk == 'LOW':
                safe_tools.append(t)



    # Prepare user message with real evidence
    evidence = {
        'job_record': job_res[0],
        'safe_tools': safe_tools
    }
    # Build evidence_sources *before* blocking on safe_tools — required by H1 contract: must always return non-empty evidence_sources/steps on PASS/BLOCKED
    evidence_sources = [
        {'tool': 'devflow.job.read', 'query': {'jobId': job_id}, 'count': len(job_res)},
        {'tool': 'devflow.catalog.search', 'query': {'query': ''}, 'count': len(catalog_res)}
    ]

    # Build steps *before* blocking on safe_tools — same reason
    steps = [
        f'从输入提取唯一任务ID：{job_id}',
        f'调用 devflow.job.read(jobId="{job_id}") → {len(job_res)} 条记录',
        f'调用 devflow.catalog.search(query="") → {len(catalog_res)} 条工具记录',
        f'筛选出 {len(safe_tools)} 个 READ/LOW 工具'
    ]

    # Block if no safe tools — required by H1: "only recommend READ/LOW", and empty list violates safety guarantee
    if not safe_tools:
        return {
            'decision': 'BLOCKED',
            'block_reason': '工具目录中无符合 READ/LOW 要求的可用工具',
            'evidence_sources': evidence_sources,
            'steps': steps
        }

    # Get prompt & compile system message
    prompt = await ctx.prompt('role')
    system_msg = prompt.compile()
    user_msg = json.dumps({'input': text, 'evidence': evidence}, ensure_ascii=False)

    # LLM decision
    try:
        llm_out = await ctx.llm.chat([
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg}
        ], prompt=prompt)
        decision_obj = json.loads(llm_out.strip())
        decision = decision_obj.get('decision')
        if decision not in ('PASS', 'BLOCKED'):
            return _blocked(f'LLM返回无效decision：{decision}')
    except Exception as e:
        return _blocked(f'LLM解析失败：{str(e)}')

    if decision == 'BLOCKED':
        return {
            'decision': 'BLOCKED',
            'block_reason': decision_obj.get('block_reason', '模型判定阻塞'),
            'evidence_sources': evidence_sources,
            'steps': steps
        }

    # Assemble answer from real evidence
    job = job_res[0]
    status = job.get('status', 'UNKNOWN').upper()
    desc_map = {
        'RUN': '运行中', 'WAIT': '等待中', 'HUMAN': '需人工介入',
        'QUEUED': '已排队', 'DONE': '已完成', 'FAIL': '执行失败', 'CANCEL': '已取消'
    }
    status_zh = desc_map.get(status, '状态未知')

    tool_names = [t.get('name', '未知工具') for t in safe_tools]
    tool_list = '、'.join(tool_names) if tool_names else '暂无符合安全要求的只读工具'

    answer = f"任务 {job_id} 当前状态：{status_zh}。可用只读工具：{tool_list}。"

    return {
        'decision': 'PASS',
        'answer': answer,
        'evidence_sources': evidence_sources,
        'steps': steps
    }


def _blocked(reason: str) -> dict:
    return {
        'decision': 'BLOCKED',
        'block_reason': reason,
        'evidence_sources': [],
        'steps': []
    }