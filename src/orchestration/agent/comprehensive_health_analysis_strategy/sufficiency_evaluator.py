# -*- coding: utf-8 -*-
"""
充分性评估模块

Qwen3结构化决策 #2：评估每维度知识充分性。

关键设计：每维度独立评估子任务，8子任务并发提交。
- prompt只含该维度的知识摘要，避免上下文过大
- 由资源池的并发控制决定实际并行度（模型并发数>1则真正并行，=1则串行）

降级策略：JSON解析失败→结构化自修复→修复失败→默认所有维度sufficient=true。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.config.business.report_service_config import get_runtime_config

logger = logging.getLogger(__name__)

_EVAL_SYSTEM_PROMPT = "你是医疗知识充分性评估器。请严格按照JSON格式输出评估结果。/no_think"


class DimensionSufficiency:
    """单维度充分性评估结果"""

    __slots__ = ("sufficient", "reason", "replace_indices")

    def __init__(
        self,
        sufficient: bool,
        reason: str = "",
        replace_indices: Optional[List[int]] = None,
    ):
        self.sufficient = sufficient
        self.reason = reason
        self.replace_indices = replace_indices or []

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"sufficient": self.sufficient}
        if self.sufficient and self.reason:
            d["reason"] = self.reason
        if not self.sufficient and self.replace_indices:
            d["replace_indices"] = self.replace_indices
        return d


class SufficiencyEvaluator:
    """
    充分性评估器

    8维度并发评估：每维度一个独立子任务，prompt只含该维度知识摘要。
    由model_service.call_model_batch()批量提交，资源池控制并发度。

    降级链：
    1. Qwen3正常返回 → 解析JSON → 验证supplement路径名
    2. JSON解析失败 → 结构化自修复（含原始prompt上下文）
    3. 自修复失败 → 规则引擎默认充分性评分
    4. Qwen3非JSON错误 → 跳过自修复，规则引擎降级
    """

    def __init__(self) -> None:
        self._config = get_runtime_config()

    def evaluate(
        self,
        dimension_knowledge: Dict[str, Any],
        dimension_used_paths: Dict[str, List[str]],
        supplement_round: int,
        model_service: Any,
    ) -> Dict[str, DimensionSufficiency]:
        """
        评估8维度知识充分性（8维度并发子任务）

        Args:
            dimension_knowledge: 维度名→DimensionKnowledge对象
            dimension_used_paths: 维度名→已用路径列表
            supplement_round: 当前补充轮次（从1开始）
            model_service: ModelBusinessService实例

        Returns:
            Dict[str, DimensionSufficiency]: 维度名→充分性评估结果
        """
        max_rounds = self._config.agent_max_iteration_rounds

        # 为每维度构建独立prompt
        prompts = {}
        for dim_name, dim_know in dimension_knowledge.items():
            hybrid_scores = None
            if hasattr(dim_know, 'hybrid_scores') and dim_know.hybrid_scores:
                hybrid_scores = dim_know.hybrid_scores
            prompt = self._build_dimension_prompt(
                dim_name, dim_know,
                dimension_used_paths.get(dim_name, []),
                supplement_round, max_rounds,
                hybrid_scores=hybrid_scores,
            )
            prompts[dim_name] = prompt

        # 通过call_model_batch批量提交（资源池控制实际并发度）
        try:
            prompt_list = [prompts[dim] for dim in dimension_knowledge]
            dim_names = list(dimension_knowledge.keys())

            batch_results = model_service.call_model_batch(
                prompt_list,
                max_tokens=self._config.agent_evaluate_max_tokens,
                timeout=self._config.agent_evaluate_timeout,
            )

            # 解析每个维度的结果
            results: Dict[str, DimensionSufficiency] = {}
            needs_repair: Dict[str, str] = {}

            for i, dim_name in enumerate(dim_names):
                raw_output = batch_results[i] if i < len(batch_results) else ""
                sufficiency = self._parse_single(raw_output, dim_name)
                if sufficiency is not None:
                    results[dim_name] = sufficiency
                else:
                    needs_repair[dim_name] = raw_output

            # 对解析失败的维度尝试自修复
            if needs_repair:
                logger.info(
                    f"[SufficiencyEvaluator] {len(needs_repair)}个维度解析失败，尝试自修复"
                )
                for dim_name, raw_output in needs_repair.items():
                    repaired = self._try_repair_single(
                        raw_output, prompts[dim_name], dim_name,
                        model_service=model_service,
                    )
                    results[dim_name] = repaired if repaired is not None else self._rule_based_score(dimension_knowledge[dim_name])

            return results

        except Exception as e:
            logger.warning(
                f"[SufficiencyEvaluator] 批量评估异常({type(e).__name__})，"
                f"降级为规则引擎评分"
            )
            return {
                dim_name: self._rule_based_score(dimension_knowledge[dim_name])
                for dim_name in dimension_knowledge
            }

    def _build_dimension_prompt(
        self,
        dim_name: str,
        dim_knowledge: Any,
        used_paths: List[str],
        supplement_round: int,
        max_rounds: int,
        hybrid_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """构建单维度充分性评估prompt"""
        truncate_len = self._config.evaluation_content_truncate_len
        knowledge_items = dim_knowledge.refined_knowledge if hasattr(dim_knowledge, 'refined_knowledge') else dim_knowledge.get("refined_knowledge", [])

        # 构建知识摘要（含混合相关性分数）
        summaries = []
        for i, item in enumerate(knowledge_items[:5]):
            content = item.get("content", item.get("description", ""))
            target = item.get("target_entity", "")
            if content:
                content = content[:truncate_len]
            score_info = ""
            if hybrid_scores and target in hybrid_scores:
                score_info = f", 相关性={hybrid_scores[target]:.2f}"
            summaries.append(f"[{i}] 目标={target}{score_info}, 内容={content}")

        knowledge_summary = "\n".join(f"- {s}" for s in summaries) if summaries else "（无知识）"
        used_paths_str = ", ".join(used_paths) if used_paths else "（无）"

        scoring_info = ""
        if hybrid_scores:
            alpha = self._config.agent_hybrid_relevance_user_weight
            beta = self._config.agent_hybrid_relevance_dim_weight
            scoring_info = (
                f"\n【混合相关性评分】公式: {alpha:.2f}×用户相关性 + {beta:.2f}×维度相关性\n"
                f"每项知识的混合相关性分数已标注在知识摘要中。\n"
            )

        return (
            f"{_EVAL_SYSTEM_PROMPT}\n\n"
            f"评估维度【{dim_name}】的知识充分性。\n\n"
            f"【当前轮次】第{supplement_round}/{max_rounds}轮\n\n"
            f"【该维度知识摘要】\n{knowledge_summary}\n\n"
            f"{scoring_info}"
            f"【已用路径】\n{used_paths_str}\n\n"
            "【约束】\n"
            "- sufficient=true时必须给出reason\n"
            "- sufficient=false时标出需要替换的知识索引(replace_indices)\n"
            "- replace_indices为知识摘要中的编号[0],[1]等\n"
            f"- 第{max_rounds}轮后强制结束，本轮请谨慎评估\n\n"
            '请输出JSON：\n'
            '{"sufficient": true/false, "reason": "..."(仅sufficient=true), '
            '"replace_indices": [0, 2](仅sufficient=false, 标出质量最低的知识索引)}\n\n'
            "/no_think"
        )

    def _parse_single(
        self,
        raw_output: str,
        dim_name: str,
    ) -> Optional[DimensionSufficiency]:
        """解析单维度评估结果"""
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

        sufficient = parsed.get("sufficient")
        if not isinstance(sufficient, bool):
            return None

        reason = parsed.get("reason", "")
        replace_indices = parsed.get("replace_indices", [])

        # 验证replace_indices为非负整数列表
        valid_indices = []
        if isinstance(replace_indices, list):
            for idx in replace_indices:
                if isinstance(idx, int) and idx >= 0:
                    valid_indices.append(idx)

        return DimensionSufficiency(
            sufficient=sufficient,
            reason=str(reason) if reason else "",
            replace_indices=valid_indices,
        )

    def _try_repair_single(
        self,
        raw_output: str,
        original_prompt: str,
        dim_name: str,
        model_service: Any = None,
    ) -> Optional[DimensionSufficiency]:
        """单维度结构化自修复"""
        try:
            if model_service is None:
                logger.warning(f"[STRUCTURED_REPAIR] model_service不可用，跳过自修复: context_type=sufficiency, dimension={dim_name}")
                return None

            repair_prompt = (
                "你上一次输出的JSON结构有误，请修复后重新输出。\n\n"
                "【错误信息】JSON解析失败\n\n"
                f"【你的原始输出】\n{raw_output}\n\n"
                '【期望格式】\n{"sufficient": true/false, '
                '"reason": "..."(仅sufficient=true), '
                '"replace_indices": [0, 2](仅sufficient=false)}\n\n'
                "【原始请求（完整上下文）】\n"
                f"{original_prompt}\n\n"
                "请直接输出修复后的JSON，不要输出其他内容。 /no_think"
            )

            messages = [
                {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ]

            logger.info(f"[STRUCTURED_REPAIR] 尝试自修复: context_type=sufficiency, dimension={dim_name}")
            repair_response = model_service.call_model(messages)

            if not repair_response:
                logger.warning(
                    f"[STRUCTURED_REPAIR] 自修复失败: 模型返回为空, "
                    f"context_type=sufficiency, dimension={dim_name}"
                )
                return None

            return self._parse_single(repair_response, dim_name)

        except Exception as e:
            logger.warning(
                f"[STRUCTURED_REPAIR] 自修复异常: error_type={type(e).__name__}, "
                f"context_type=sufficiency, dimension={dim_name}"
            )
            return None

    def _rule_based_score(self, dim_knowledge: Any) -> DimensionSufficiency:
        """规则引擎默认充分性评分（降级策略Level 2）"""
        knowledge_items = (
            dim_knowledge.refined_knowledge
            if hasattr(dim_knowledge, 'refined_knowledge')
            else dim_knowledge.get("refined_knowledge", [])
        )
        count = len(knowledge_items)
        if count >= 5:
            return DimensionSufficiency(sufficient=True, reason="规则引擎: 知识充分(>=5条)")
        elif count >= 3:
            return DimensionSufficiency(sufficient=True, reason="规则引擎: 知识基本充分(>=3条)")
        elif count >= 1:
            return DimensionSufficiency(sufficient=False, reason="规则引擎: 知识不足(<3条)")
        else:
            return DimensionSufficiency(sufficient=False, reason="规则引擎: 无知识(0条)")

    def evaluate_rule_based(self, dim_knowledge: Any) -> DimensionSufficiency:
        """公共接口：规则引擎充分性评分"""
        return self._rule_based_score(dim_knowledge)
