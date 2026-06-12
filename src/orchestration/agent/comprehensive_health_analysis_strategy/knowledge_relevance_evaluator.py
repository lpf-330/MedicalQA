# -*- coding: utf-8 -*-
"""
混合相关性评估模块

Qwen3结构化决策：对每个维度的每项知识打分（user_relevance + dimension_relevance + is_core），
程序计算混合分数 mixed_score = α·user_relevance + β·dimension_relevance。

位于跨维度去重之后、充分性评估之前。
评分结果写入DimensionKnowledge.hybrid_scores，供EvaluateSufficiency参考。

降级链：
1. Qwen3返回 → 解析每项评分 → 计算mixed_score
2. JSON解析失败 → 结构化自修复
3. 自修复失败 → 默认分数0.5
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.config.business.report_service_config import get_runtime_config

logger = logging.getLogger(__name__)

_RELEVANCE_SYSTEM_PROMPT = "你是医学知识相关性评估器。请严格按照JSON格式输出评估结果。/no_think"


class KnowledgeRelevanceEvaluator:
    """
    混合相关性评估器

    Qwen3为每项知识打分，程序计算混合分数。
    公式：mixed_score = α·user_relevance + β·dimension_relevance + γ·knowledge_quality
    α/β/γ通过配置参数控制。
    """

    def evaluate(
        self,
        dimension_knowledge: Dict[str, Any],
        user_info: str,
        model_service: Any,
    ) -> Dict[str, Dict[str, float]]:
        """
        8维度并发评估知识相关性

        Args:
            dimension_knowledge: 维度名→DimensionKnowledge对象
            user_info: 用户信息摘要
            model_service: ModelBusinessService实例

        Returns:
            Dict[str, Dict[str, float]]: 维度名→{target_entity: mixed_score}
        """
        config = get_runtime_config()
        alpha = config.relevance_alpha
        beta = config.relevance_beta
        gamma = config.relevance_gamma

        # 为每维度构建独立prompt
        prompts = {}
        dim_knowledge_map = {}
        for dim_name, dim_know in dimension_knowledge.items():
            knowledge_items = dim_know.refined_knowledge if hasattr(dim_know, 'refined_knowledge') else dim_know.get("refined_knowledge", [])
            if not knowledge_items:
                continue
            prompt = self._build_dimension_prompt(
                dim_name, knowledge_items, user_info, config
            )
            prompts[dim_name] = prompt
            dim_knowledge_map[dim_name] = knowledge_items

        if not prompts:
            logger.info("[HybridRelevance] 无维度需要评估")
            return {}

        # 批量提交
        try:
            prompt_list = [prompts[dim] for dim in prompts]
            dim_names = list(prompts.keys())

            batch_results = model_service.call_model_batch(
                prompt_list,
                max_tokens=config.agent_hybrid_relevance_max_tokens,
                timeout=config.batch_evaluation_timeout,
            )

            # 解析并计算混合分数
            all_scores: Dict[str, Dict[str, float]] = {}
            needs_repair: Dict[str, str] = {}

            for i, dim_name in enumerate(dim_names):
                raw_output = batch_results[i] if i < len(batch_results) else ""
                knowledge_items = dim_knowledge_map[dim_name]
                dim_scores = self._parse_single(raw_output, dim_name, knowledge_items, alpha, beta, gamma)
                if dim_scores is not None:
                    all_scores[dim_name] = dim_scores
                else:
                    needs_repair[dim_name] = raw_output

            # 自修复
            if needs_repair:
                logger.info(
                    f"[HybridRelevance] {len(needs_repair)}个维度解析失败，尝试自修复"
                )
                for dim_name, raw_output in needs_repair.items():
                    repaired = self._try_repair_single(
                        raw_output, prompts[dim_name], dim_name,
                        dim_knowledge_map[dim_name], alpha, beta, gamma,
                        model_service, config,
                    )
                    if repaired is not None:
                        all_scores[dim_name] = repaired
                    else:
                        # 降级：默认0.5
                        items = dim_knowledge_map[dim_name]
                        all_scores[dim_name] = {
                            item.get("target_entity", f"item_{j}"): 0.5
                            for j, item in enumerate(items)
                        }

            # 写入DimensionKnowledge.hybrid_scores
            for dim_name, scores in all_scores.items():
                if dim_name in dimension_knowledge:
                    dim_know = dimension_knowledge[dim_name]
                    if hasattr(dim_know, 'hybrid_scores'):
                        dim_know.hybrid_scores = scores

            total_scored = sum(len(v) for v in all_scores.values())
            logger.info(
                f"[HybridRelevance] 评估完成: 维度数={len(all_scores)}, "
                f"总评分项={total_scored}"
            )

            return all_scores

        except Exception as e:
            logger.warning(
                f"[HybridRelevance] 批量评估异常({type(e).__name__})，"
                f"降级为默认分数0.5"
            )
            fallback: Dict[str, Dict[str, float]] = {}
            for dim_name, dim_know in dimension_knowledge.items():
                knowledge_items = dim_know.refined_knowledge if hasattr(dim_know, 'refined_knowledge') else []
                fallback[dim_name] = {
                    item.get("target_entity", f"item_{j}"): 0.5
                    for j, item in enumerate(knowledge_items)
                }
                if hasattr(dim_know, 'hybrid_scores'):
                    dim_know.hybrid_scores = fallback[dim_name]
            return fallback

    def _build_dimension_prompt(
        self,
        dim_name: str,
        knowledge_items: List[Dict],
        user_info: str,
        config: Any,
    ) -> str:
        truncate_len = config.agent_hybrid_relevance_content_truncate_len

        items_text = []
        for i, item in enumerate(knowledge_items[:10]):
            target = item.get("target_entity", item.get("entity_name", ""))
            content = item.get("content", "")
            if content and len(content) > truncate_len:
                content = content[:truncate_len] + "..."
            items_text.append(f"[{i}] 目标={target}, 内容={content}")

        candidates_text = "\n".join(items_text)

        return (
            f"{_RELEVANCE_SYSTEM_PROMPT}\n\n"
            f"评估维度【{dim_name}】中每项知识的相关性。\n\n"
            f"【用户信息】\n{user_info[:500]}\n\n"
            f"【知识列表】\n{candidates_text}\n\n"
            f"【评分标准】\n"
            f"- user_relevance (0-1): 与用户具体情况的相关程度\n"
            f"- dimension_relevance (0-1): 对该维度知识需求的满足程度\n"
            f"- knowledge_quality (0-1): 知识本身的质量和可信度\n"
            f"- is_core (bool): 是否为核心知识（必须保留）\n\n"
            f'请输出JSON：{{"items": [{{"index": 0, "user_relevance": 0.8, "dimension_relevance": 0.7, "knowledge_quality": 0.9, "is_core": true}}, ...]}}\n\n'
            f"/no_think"
        )

    def _parse_single(
        self,
        raw_output: str,
        dim_name: str,
        knowledge_items: List[Dict],
        alpha: float,
        beta: float,
        gamma: float,
    ) -> Optional[Dict[str, float]]:
        """解析单维度评估结果，计算mixed_score"""
        if not raw_output:
            return None

        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if not json_match:
            return None

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        items = parsed.get("items")
        if not isinstance(items, list):
            return None

        scores: Dict[str, float] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            user_rel = item.get("user_relevance", 0.5)
            dim_rel = item.get("dimension_relevance", 0.5)
            kq = item.get("knowledge_quality", 0.5)

            if not isinstance(user_rel, (int, float)):
                user_rel = 0.5
            if not isinstance(dim_rel, (int, float)):
                dim_rel = 0.5
            if not isinstance(kq, (int, float)):
                kq = 0.5

            user_rel = max(0.0, min(1.0, float(user_rel)))
            dim_rel = max(0.0, min(1.0, float(dim_rel)))
            kq = max(0.0, min(1.0, float(kq)))

            mixed_score = max(0.0, min(1.0, alpha * user_rel + beta * dim_rel + gamma * kq))

            # 用target_entity作为key，回退到item_{idx}
            if isinstance(idx, int) and 0 <= idx < len(knowledge_items):
                target = knowledge_items[idx].get("target_entity", f"item_{idx}")
            else:
                target = f"item_{idx}"
            scores[target] = round(mixed_score, 4)

        return scores if scores else None

    def _try_repair_single(
        self,
        raw_output: str,
        original_prompt: str,
        dim_name: str,
        knowledge_items: List[Dict],
        alpha: float,
        beta: float,
        gamma: float,
        model_service: Any,
        config: Any,
    ) -> Optional[Dict[str, float]]:
        """单维度结构化自修复"""
        try:
            if model_service is None:
                logger.warning(f"[STRUCTURED_REPAIR] model_service不可用，跳过自修复: context_type=hybrid_relevance, dimension={dim_name}")
                return None

            repair_prompt = (
                "你上一次输出的JSON结构有误，请修复后重新输出。\n\n"
                "【错误信息】JSON解析失败\n\n"
                f"【你的原始输出】\n{raw_output}\n\n"
                '【期望格式】\n{"items": [{"index": 0, "user_relevance": 0.8, '
                '"dimension_relevance": 0.7, "knowledge_quality": 0.9, "is_core": true}, ...]}\n\n'
                "【原始请求（完整上下文）】\n"
                f"{original_prompt}\n\n"
                "请直接输出修复后的JSON，不要输出其他内容。 /no_think"
            )

            messages = [
                {"role": "system", "content": _RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ]

            logger.info(f"[STRUCTURED_REPAIR] 尝试自修复: context_type=hybrid_relevance, dimension={dim_name}")
            repair_response = model_service.call_model(messages)

            if not repair_response:
                return None

            return self._parse_single(repair_response, dim_name, knowledge_items, alpha, beta, gamma)

        except Exception as e:
            logger.warning(
                f"[STRUCTURED_REPAIR] 自修复异常: error_type={type(e).__name__}, "
                f"context_type=hybrid_relevance, dimension={dim_name}"
            )
            return None
