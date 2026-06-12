# -*- coding: utf-8 -*-
"""
检索规划模块

Qwen3结构化决策 #1：为每维度选择检索路径和关注实体。

职责边界：只做路径选择和实体关联，不做检索、不做内容判断、不做充分性评估。
降级策略：JSON解析失败→结构化自修复→修复失败→降级到DIMENSION_RECOMMENDATIONS推荐路径。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.config.business.report_service_config import get_runtime_config
from src.orchestration.agent.comprehensive_health_analysis_strategy.path_registry import (
    get_available_paths_summary,
    get_dimension_recommendations_summary,
    get_recommended_paths_for_dimension,
    validate_path_name,
)

logger = logging.getLogger(__name__)

_PLAN_SYSTEM_PROMPT = "你是医疗知识检索规划器。请严格按照JSON格式输出检索路径规划。/no_think"


class RetrievalPlan:
    """单维度的检索规划结果"""

    __slots__ = ("paths", "entities")

    def __init__(self, paths: List[str], entities: List[str]):
        self.paths = paths
        self.entities = entities

    def to_dict(self) -> Dict[str, Any]:
        return {"paths": self.paths, "entities": self.entities}


class RetrievalPlanner:
    """
    检索规划器

    调用Qwen3为8维度选择检索路径。Qwen3只做结构化决策，不做实际检索。

    降级链：
    1. Qwen3正常返回 → 解析JSON → 验证路径名
    2. JSON解析失败 → 结构化自修复（含原始prompt上下文）→ 重试解析
    3. 自修复失败 → 降级到DIMENSION_RECOMMENDATIONS推荐路径
    4. Qwen3非JSON错误（超时/连接失败）→ 跳过自修复，直接降级
    """

    def __init__(self) -> None:
        self._config = get_runtime_config()

    def plan(
        self,
        dimension_queries: Dict[str, str],
        ner_entities: Dict[str, List[str]],
        model_service: Any = None,
    ) -> Dict[str, RetrievalPlan]:
        """
        为8维度生成检索规划

        Args:
            dimension_queries: 维度名→查询文本
            ner_entities: NER实体分类，如 {"disease_names": [...], "symptom_names": [...]}
            model_service: ModelBusinessService实例

        Returns:
            Dict[str, RetrievalPlan]: 维度名→检索规划
        """
        max_paths = self._config.agent_max_paths_per_dimension
        prompt = self._build_prompt(dimension_queries, ner_entities, max_paths)

        try:
            if model_service is None:
                logger.warning("[RetrievalPlanner] model_service不可用，降级到推荐路径")
                return self._fallback_plan(dimension_queries, max_paths)

            messages = [
                {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            logger.info("[LLM_INPUT] 检索路径规划, prompt_len=%d", len(prompt))
            raw_output = model_service.call_model(messages)

            if raw_output:
                logger.info("[LLM_OUTPUT] 检索路径规划, response_len=%d", len(raw_output))

            if not raw_output:
                logger.warning("[RetrievalPlanner] Qwen3返回为空，降级到推荐路径")
                return self._fallback_plan(dimension_queries, max_paths)

            plan_result = self._parse_and_validate(raw_output, dimension_queries, max_paths)
            if plan_result is not None:
                return plan_result

            # JSON解析失败，尝试自修复
            logger.info("[RetrievalPlanner] JSON解析失败，尝试结构化自修复")
            plan_result = self._try_repair(raw_output, prompt, dimension_queries, max_paths, model_service=model_service)
            if plan_result is not None:
                return plan_result

            # 自修复也失败，降级
            logger.warning("[RetrievalPlanner] 自修复失败，降级到推荐路径")
            return self._fallback_plan(dimension_queries, max_paths)

        except Exception as e:
            logger.warning(
                f"[RetrievalPlanner] Qwen3调用异常({type(e).__name__})，"
                f"跳过自修复，降级到推荐路径"
            )
            return self._fallback_plan(dimension_queries, max_paths)

    def _build_prompt(
        self,
        dimension_queries: Dict[str, str],
        ner_entities: Dict[str, List[str]],
        max_paths: int,
    ) -> str:
        """构建PlanRetrieval prompt"""
        paths_summary = get_available_paths_summary()
        dim_summary = get_dimension_recommendations_summary()

        entities_str = "\n".join(
            f"  {k}: {', '.join(v[:5])}" for k, v in ner_entities.items() if v
        )

        queries_str = "\n".join(
            f"  {dim}: {query[:80]}" for dim, query in dimension_queries.items()
        )

        return (
            "根据以下信息，为每个维度选择检索路径。\n\n"
            f"【维度推荐路径】\n{dim_summary}\n\n"
            f"【可选路径】\n{paths_summary}\n\n"
            f"【NER识别实体】\n{entities_str}\n\n"
            f"【8维度查询文本】\n{queries_str}\n\n"
            f"【约束】\n"
            f"- 每维度最多选择{max_paths}条路径\n"
            "- 路径名必须来自可选路径列表\n"
            "- entities字段填写该维度应关注的实体名称\n\n"
            '请输出JSON，格式：\n'
            '{{"维度名": {{"paths": ["路径名"], "entities": ["实体名"]}}}}\n\n'
            "/no_think"
        )

    def _parse_and_validate(
        self,
        raw_output: str,
        dimension_queries: Dict[str, str],
        max_paths: int,
    ) -> Optional[Dict[str, RetrievalPlan]]:
        """解析并验证Qwen3输出"""
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if not json_match:
            return None

        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        plan_result: Dict[str, RetrievalPlan] = {}
        for dim_name in dimension_queries:
            dim_plan = parsed.get(dim_name, {})
            if not isinstance(dim_plan, dict):
                dim_plan = {}

            paths = dim_plan.get("paths", [])
            entities = dim_plan.get("entities", [])

            valid_paths = [p for p in paths if validate_path_name(p)][:max_paths]
            if not valid_paths:
                valid_paths = get_recommended_paths_for_dimension(dim_name)[:max_paths]

            plan_result[dim_name] = RetrievalPlan(paths=valid_paths, entities=entities)

        return plan_result

    def _try_repair(
        self,
        raw_output: str,
        original_prompt: str,
        dimension_queries: Dict[str, str],
        max_paths: int,
        model_service: Any = None,
    ) -> Optional[Dict[str, RetrievalPlan]]:
        """结构化自修复：含原始prompt上下文"""
        try:
            if model_service is None:
                logger.warning("[STRUCTURED_REPAIR] model_service不可用，跳过自修复")
                return None

            repair_prompt = (
                "你上一次输出的JSON结构有误，请修复后重新输出。\n\n"
                "【错误信息】JSON解析失败\n\n"
                f"【你的原始输出】\n{raw_output}\n\n"
                "【期望格式】\n"
                '{"维度名": {"paths": ["路径名"], "entities": ["实体名"]}}\n\n'
                "【原始请求（完整上下文）】\n"
                f"{original_prompt}\n\n"
                "请直接输出修复后的JSON，不要输出其他内容。 /no_think"
            )

            messages = [
                {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ]

            logger.info("[STRUCTURED_REPAIR] 尝试自修复: context_type=retrieval_plan")
            repair_response = model_service.call_model(messages)

            if not repair_response:
                logger.warning("[STRUCTURED_REPAIR] 自修复失败: 模型返回为空, context_type=retrieval_plan")
                return None

            logger.info(
                f"[STRUCTURED_REPAIR_OUTPUT] context_type=retrieval_plan, "
                f"response_len={len(repair_response)}"
            )
            return self._parse_and_validate(repair_response, dimension_queries, max_paths)

        except Exception as e:
            logger.warning(
                f"[STRUCTURED_REPAIR] 自修复异常: error_type={type(e).__name__}, "
                f"context_type=retrieval_plan"
            )
            return None

    def _fallback_plan(
        self,
        dimension_queries: Dict[str, str],
        max_paths: int,
    ) -> Dict[str, RetrievalPlan]:
        """降级到DIMENSION_RECOMMENDATIONS推荐路径"""
        logger.info("[RetrievalPlanner] 使用DIMENSION_RECOMMENDATIONS推荐路径")
        plan_result: Dict[str, RetrievalPlan] = {}
        for dim_name in dimension_queries:
            rec_paths = get_recommended_paths_for_dimension(dim_name)[:max_paths]
            plan_result[dim_name] = RetrievalPlan(paths=rec_paths, entities=[])
        return plan_result
