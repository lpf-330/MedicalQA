# -*- coding: utf-8 -*-
"""
知识面选取模块

Qwen3结构化决策：当候选知识数量超过limit时，由Qwen3选取最佳知识面覆盖，
替代硬截断[:limit]，解决多疾病场景下知识覆盖不均的问题。

降级链：
1. Qwen3返回 → 解析selected_indices → 验证索引范围
2. JSON解析失败 → 结构化自修复
3. 自修复失败 → 降级为[:limit]硬截断
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from src.config.business.report_service_config import get_runtime_config

logger = logging.getLogger(__name__)

_SELECTION_SYSTEM_PROMPT = "你是医学知识面选取器。请严格按照JSON格式输出选取结果。/no_think"


class KnowledgeSelector:
    """
    知识面选取器

    当维度候选知识超过limit时，Qwen3根据用户信息和维度需求选取最佳覆盖。
    输入：用户信息 + 维度名 + 候选知识摘要 + 黑名单 + limit
    输出：{"selected_indices": [0, 3, 5]}  (候选列表中的索引)
    """

    def select(
        self,
        dim_name: str,
        candidates: List[Dict],
        limit: int,
        user_info: str,
        blacklist: Set[str],
        model_service: Any,
    ) -> List[Dict]:
        """
        对候选知识进行知识面选取

        Args:
            dim_name: 维度名
            candidates: 候选知识列表
            limit: 目标选取数量
            user_info: 用户信息摘要
            blacklist: 知识黑名单（target_entity集合）
            model_service: ModelBusinessService实例

        Returns:
            选取后的知识列表（不超过limit条）
        """
        config = get_runtime_config()

        # 先过滤黑名单
        filtered = [
            item for item in candidates
            if item.get("target_entity", item.get("entity_name", "")) not in blacklist
        ]

        if len(filtered) <= limit:
            logger.info(
                f"[KnowledgeSelector] 维度{dim_name}: "
                f"候选数={len(filtered)}(过滤黑名单后) <= limit={limit}, 无需选取"
            )
            return filtered

        # Qwen3知识面选取
        prompt = self._build_prompt(
            dim_name, filtered, limit, user_info, config
        )

        raw_output = ""
        try:
            messages = [
                {"role": "system", "content": _SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            raw_output = model_service.call_model(
                messages,
                timeout=config.agent_evaluate_timeout,
            )

            selected_indices = self._parse_result(raw_output, len(filtered), dim_name)

            if selected_indices is not None:
                selected = [filtered[i] for i in selected_indices]
                logger.info(
                    f"[KnowledgeSelector] 维度{dim_name}: "
                    f"Qwen3选取={len(selected)}/{len(filtered)}, "
                    f"indices={selected_indices}"
                )
                return selected

        except Exception as e:
            logger.warning(
                f"[KnowledgeSelector] 维度{dim_name}: "
                f"Qwen3选取异常({type(e).__name__})，尝试自修复"
            )

        # 尝试自修复
        if raw_output:
            try:
                repaired = self._try_repair(raw_output, prompt, len(filtered), dim_name, model_service, config)
                if repaired is not None:
                    selected = [filtered[i] for i in repaired]
                    logger.info(
                        f"[KnowledgeSelector] 维度{dim_name}: "
                        f"自修复选取={len(selected)}/{len(filtered)}"
                    )
                    return selected
            except Exception as e:
                logger.warning(
                    f"[KnowledgeSelector] 维度{dim_name}: "
                    f"自修复异常({type(e).__name__})，降级为硬截断"
                )

        # 降级：硬截断
        logger.warning(
            f"[KnowledgeSelector] 维度{dim_name}: "
            f"降级为硬截断[:{limit}]"
        )
        return filtered[:limit]

    def _build_prompt(
        self,
        dim_name: str,
        candidates: List[Dict],
        limit: int,
        user_info: str,
        config: Any,
    ) -> str:
        truncate_len = config.agent_knowledge_selection_content_truncate_len

        items_text = []
        for i, item in enumerate(candidates):
            source = item.get("source_entity", "")
            relation = item.get("relation_type", "")
            target = item.get("target_entity", item.get("entity_name", ""))
            content = item.get("content", "")
            if content and len(content) > truncate_len:
                content = content[:truncate_len] + "..."
            items_text.append(f"[{i}] 来源={source}, 关系={relation}, 目标={target}, 内容={content}")

        candidates_text = "\n".join(items_text)

        return (
            f"{_SELECTION_SYSTEM_PROMPT}\n\n"
            f"为维度【{dim_name}】从候选知识中选取最佳覆盖。\n\n"
            f"【用户信息】\n{user_info[:500]}\n\n"
            f"【候选知识】(共{len(candidates)}条)\n{candidates_text}\n\n"
            f"【选取要求】\n"
            f"- 从上述候选中选取{limit}条知识\n"
            f"- 确保覆盖不同疾病/实体，避免只选同一疾病的知识\n"
            f"- 优先选取与用户情况最相关的知识\n"
            f"- 优先选取信息量更丰富的知识\n\n"
            f'请输出JSON：{{"selected_indices": [索引列表]}}\n\n'
            f"/no_think"
        )

    def _parse_result(
        self,
        raw_output: str,
        candidate_count: int,
        dim_name: str,
    ) -> Optional[List[int]]:
        """解析Qwen3选取结果"""
        if not raw_output:
            return None

        json_match = re.search(r'\{[^{}]*\}', raw_output)
        if not json_match:
            return None

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        indices = parsed.get("selected_indices")
        if not isinstance(indices, list):
            return None

        valid_indices = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < candidate_count:
                valid_indices.append(idx)

        if not valid_indices:
            return None

        # 去重
        seen = set()
        deduped = []
        for idx in valid_indices:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)

        return deduped

    def _try_repair(
        self,
        raw_output: str,
        original_prompt: str,
        candidate_count: int,
        dim_name: str,
        model_service: Any,
        config: Any,
    ) -> Optional[List[int]]:
        """结构化自修复"""
        try:
            if model_service is None:
                logger.warning(f"[STRUCTURED_REPAIR] model_service不可用，跳过自修复: context_type=knowledge_selection, dimension={dim_name}")
                return None

            repair_prompt = (
                "你上一次输出的JSON结构有误，请修复后重新输出。\n\n"
                "【错误信息】JSON解析失败\n\n"
                f"【你的原始输出】\n{raw_output}\n\n"
                '【期望格式】\n{"selected_indices": [0, 1, 2]}\n\n'
                "【原始请求（完整上下文）】\n"
                f"{original_prompt}\n\n"
                "请直接输出修复后的JSON，不要输出其他内容。 /no_think"
            )

            messages = [
                {"role": "system", "content": _SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ]

            logger.info(f"[STRUCTURED_REPAIR] 尝试自修复: context_type=knowledge_selection, dimension={dim_name}")
            repair_response = model_service.call_model(messages)

            if not repair_response:
                return None

            return self._parse_result(repair_response, candidate_count, dim_name)

        except Exception as e:
            logger.warning(
                f"[STRUCTURED_REPAIR] 自修复异常: error_type={type(e).__name__}, "
                f"context_type=knowledge_selection, dimension={dim_name}"
            )
            return None
