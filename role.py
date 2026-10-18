import json
import re
from typing import Dict, Any, List, Optional


async def execute(data: dict, ctx) -> dict:
    text = data.get('text', '').strip()
    if not text:
        return _blocked('输入为空', evidence_sources=[], steps=[])

    # Extract exactly one DF-xxx task ID: support Chinese adjacent (e.g., "任务DF-0012。" or "DF-0012状态？")
    # H1: Chinese紧邻编号 → allow any Unicode word boundary; r'DF-\d+' is sufficient
    df_matches = re.findall(r'DF-\d+', text)
    if len(df_matches) != 1:
        return _blocked(f'必须且仅能指定一个DF编号，当前匹配到{len(df_matches)}个：{df_matches}', evidence_sources=[], steps=[])
    job_id = df_matches[0]

    # Call tools
    job_res = None
    catalog_res = None
    evidence_sources = []
    steps = []

    try:
        job_res = await ctx.tools.call('devflow.job.read', jobId=job_id)
        if not isinstance(job_res, list):
            return _blocked('devflow.job.read 返回非列表类型', evidence_sources=evidence_sources, steps=steps)
        if not job_res:
            return _blocked(f'未查到任务 {job_id} 的任何记录', evidence_sources=evidence_sources, steps=steps)
        # Validate job record structure: must be dict with 'status'
        first_job = job_res[0]
        if not isinstance(first_job, dict) or 'status' not in first_job:
            return _blocked(f'任务 {job_id} 记录格式无效：缺少 status 字段', evidence_sources=evidence_sources, steps=steps)
        
        evidence_sources.append({'tool': 'devflow.job.read', 'query': {'jobId': job_id}, 'count': len(job_res)})
        steps.append(f'从输入提取唯一任务ID：{job_id}')
        steps.append(f'调用 devflow.job.read(jobId="{job_id}") → {len(job_res)} 条记录')

    except Exception as e:
        return _blocked(f'工具 devflow.job.read 调用异常：{str(e)}', evidence_sources=evidence_sources, steps=steps)

    try:
        catalog_res = await ctx.tools.call('devflow.catalog.search', query='')
        if not isinstance(catalog_res, list):
            return _blocked('devflow.catalog.search 返回非列表类型', evidence_sources=evidence_sources, steps=steps)
        if not catalog_res:
            return _blocked('工具目录为空，无可用只读工具', evidence_sources=evidence_sources, steps=steps)
        
        evidence_sources.append({'tool': 'devflow.catalog.search', 'query': {'query': ''}, 'count': len(catalog_res)})
        steps.append(f'调用 devflow.catalog.search(query="") → {len(catalog_res)} 条工具记录')

    except Exception as e:
        return _blocked(f'工具 devflow.catalog.search 调用异常：{str(e)}', evidence_sources=evidence_sources, steps=steps)

    # Filter only READ + LOW tools using dict.get() and case-insensitive normalize (per contract)
    safe_tools = []
    for t in catalog_res:
        if isinstance(t, dict):
            access = str(t.get('access', '')).strip().upper()
            risk = str(t.get('risk', '')).strip().upper()
            if access == 'READ' and risk == 'LOW':
                safe_tools.append(t)
    
    steps.append(f'筛选出 {len(safe_tools)} 个 READ/LOW 工具')

    # Block if no safe tools
    if not safe_tools:
        return _blocked('工具目录中无符合 READ/LOW 要求的可用工具', evidence_sources=evidence_sources, steps=steps)

    # Prepare user message with real evidence
    evidence = {
        'job_record': job_res[0],
        'safe_tools': safe_tools
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
        # Parse as JSON; allow optional answer/block_reason per H1
        try:
            decision_obj = json.loads(llm_out.strip())
        except json.JSONDecodeError as e:
            return _blocked(f'LLM返回非法JSON：{str(e)}', evidence_sources=evidence_sources, steps=steps)
        
        # decision is mandatory
        decision = decision_obj.get('decision')
        if decision not in ('PASS', 'BLOCKED'):
            return _blocked(f'LLM返回无效decision：{decision}', evidence_sources=evidence_sources, steps=steps)
        
        # If PASS, answer is optional but must be string if present
        if decision == 'PASS':
            answer = decision_obj.get('answer')
            if answer is not None and not isinstance(answer, str):
                return _blocked(f'LLM返回的answer字段必须为字符串，实际类型：{type(answer).__name__}', evidence_sources=evidence_sources, steps=steps)
                
    except Exception as e:
        return _blocked(f'LLM调用或解析异常：{str(e)}', evidence_sources=evidence_sources, steps=steps)

    if decision == 'BLOCKED':
        block_reason = decision_obj.get('block_reason')
        if block_reason is not None and not isinstance(block_reason, str):
            block_reason = '模型返回的block_reason字段类型非法'
        return {
            'decision': 'BLOCKED',
            'block_reason': block_reason or '模型判定阻塞',
            'evidence_sources': evidence_sources,
            'steps': steps
        }

    # Assemble answer from real evidence (fallback if model didn't provide valid answer)
    job = job_res[0]
    status = job.get('status', 'UNKNOWN').upper()
    desc_map = {
        'RUN': '运行中', 'WAIT': '等待中', 'HUMAN': '需人工介入',
        'QUEUED': '已排队', 'DONE': '已完成', 'FAIL': '执行失败', 'CANCEL': '已取消'
    }
    status_zh = desc_map.get(status, '状态未知')

    tool_names = [t.get('name', '未知工具') for t in safe_tools]
    tool_list = '、'.join(tool_names) if tool_names else '暂无符合安全要求的只读工具'

    # Use model's answer if valid, otherwise construct
    answer = decision_obj.get('answer')
    if not isinstance(answer, str):
        answer = f"任务 {job_id} 当前状态：{status_zh}。可用只读工具：{tool_list}。"

    return {
        'decision': 'PASS',
        'answer': answer,
        'evidence_sources': evidence_sources,
        'steps': steps
    }


def _blocked(reason: str, evidence_sources: List[Dict], steps: List[str]) -> dict:
    return {
        'decision': 'BLOCKED',
        'block_reason': reason,
        'evidence_sources': evidence_sources,
        'steps': steps
    }