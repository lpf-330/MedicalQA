# -*- coding: utf-8 -*-
"""
补丁路径规划模块

Qwen3结构化决策：为空位维度规划补充检索路径（轻量级决策）。
与PlanRetrieval的区别：只针对不充分维度，输入更聚焦。

降级链：
1. Qwen3返回 → 解析路径规划 → 验证路径名
2. JSON解析失败 → 结构化自修复
3. 自修复失败 → 降级为DIMENSION_RECOMMENDATIONS.supplement_paths
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from src.config.business.report_service_config import get_runtime_config
from src.orchestration.agent.comprehensive_health_analysis_strategy.path_registry import (
    get_supplement_paths_for_dimension,
    validate_path_name,
    get_available_paths_summary,
)

logger = logging.getLogger(__name__)

_PATCH_SYSTEM_PROMPT = "你是医疗检索路径规划器。请严格按照JSON格式输出规划结果。/no_think"


class PatchPathPlanner:
    """
    补丁路径规划器

    为EvaluateSufficiency判定不充分的维度规划补充检索路径。
    轻量级决策：只涉及空位维度，不涉及全部8维度。
    """

    def plan(
        self,
        vacancy_dimensions: List[str],
        dimension_used_paths: Dict[str, List[str]],
        retained_knowledge: Dict[str, List[Dict]],
        blacklist_entities: Set[str],
        user_info: str,
        model_service: Any,
    ) -> Dict[str, List[str]]:
        """
        为空位维度规划补充检索路径

        Args:
            vacancy_dimensions: 需要补充的维度列表
            dimension_used_paths: 维度名→已用路径列表
            retained_knowledge: 维度名→保留的知识条目
            blacklist_entities: 黑名单实体集合
            user_info: 用户信息摘要
            model_service: ModelBusinessService实例

        Returns:
            Dict[str, List[str]]: 维度名→补充路径列表
        """
        if not vacancy_dimensions:
            return {}

        config = get_runtime_config()

        prompt = self._build_prompt(
            vacancy_dimensions, dimension_used_paths,
            retained_knowledge, user_info, config,
        )

        raw_output = ""
        try:
            messages = [
                {"role": "system", "content": _PATCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            raw_output = model_service.call_model(
                messages,
                timeout=config.agent_evaluate_timeout,
            )

            parsed = self._parse_result(raw_output, vacancy_dimensions)

            if parsed is not None:
                logger.info(
                    f"[PatchPathPlan] 规划完成: 空位维度={vacancy_dimensions}, "
                    f"规划={parsed}"
                )
                return parsed

        except Exception as e:
            logger.warning(
                f"[PatchPathPlan] Qwen3规划异常({type(e).__name__})，尝试自修复"
            )

        # 尝试自修复
        if raw_output:
            try:
                repaired = self._try_repair(
                    raw_output, prompt, vacancy_dimensions, model_service, config,
                )
                if repaired is not None:
                    logger.info(
                        f"[PatchPathPlan] 自修复规划完成: {repaired}"
                    )
                    return repaired
            except Exception as e:
                logger.warning(
                    f"[PatchPathPlan] 自修复异常({type(e).__name__})，降级为推荐补充路径"
                )

        # 降级：使用DIMENSION_RECOMMENDATIONS.supplement_paths
        fallback = {}
        for dim in vacancy_dimensions:
            supp = get_supplement_paths_for_dimension(dim)
            used = set(dimension_used_paths.get(dim, []))
            new_paths = [p for p in supp if p not in used]
            if new_paths:
                fallback[dim] = new_paths
            else:
                fallback[dim] = supp

        logger.info(
            f"[PatchPathPlan] 降级为推荐补充路径: {fallback}"
        )
        return fallback

    def _build_prompt(
        self,
        vacancy_dimensions: List[str],
        dimension_used_paths: Dict[str, List[str]],
        retained_knowledge: Dict[str, List[Dict]],
        user_info: str,
        config: Any,
    ) -> str:
        paths_summary = get_available_paths_summary()

        dim_descriptions = []
        for dim in vacancy_dimensions:
            used = dimension_used_paths.get(dim, [])
            retained = retained_knowledge.get(dim, [])
            retained_summary = "; ".join(
                item.get("target_entity", "")[:50]
                for item in retained[:3]
                if item.get("target_entity")
            )
            dim_descriptions.append(
                f"- {dim}: 已用路径=[{', '.join(used)}], "
                f"保留知识目标=[{retained_summary}]"
            )

        dims_text = "\n".join(dim_descriptions)

        return (
            f"{_PATCH_SYSTEM_PROMPT}\n\n"
            f"为以下空位维度规划补充检索路径。\n\n"
            f"【用户信息】\n{user_info[:400]}\n\n"
            f"【空位维度信息】\n{dims_text}\n\n"
            f"【可用路径】\n{paths_summary}\n\n"
            f"【约束】\n"
            f"- 只规划空位维度，不需要规划已充分维度\n"
            f"- 路径不能与已用路径重复\n"
            f"- 每个维度最多2条补充路径\n"
            f"- 路径名必须来自可用路径列表\n\n"
            f'请输出JSON：{{"dimensions": {{"维度名": ["路径1", "路径2"], ...}}}}\n\n'
            f"/no_think"
        )

    def _parse_result(
        self,
        raw_output: str,
        vacancy_dimensions: List[str],
    ) -> Optional[Dict[str, List[str]]]:
        """解析Qwen3路径规划结果"""
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

        dimensions = parsed.get("dimensions", parsed)

        if not isinstance(dimensions, dict):
            return None

        result: Dict[str, List[str]] = {}
        for dim_name, paths in dimensions.items():
            if dim_name not in vacancy_dimensions:
                continue
            if not isinstance(paths, list):
                continue
            valid_paths = [
                p for p in paths
                if isinstance(p, str) and validate_path_name(p)
            ]
            if valid_paths:
                result[dim_name] = valid_paths[:2]

        return result if result else None

    def _try_repair(
        self,
        raw_output: str,
        original_prompt: str,
        vacancy_dimensions: List[str],
        model_service: Any,
        config: Any,
    ) -> Optional[Dict[str, List[str]]]:
        """结构化自修复"""
        try:
            if model_service is None:
                logger.warning("[STRUCTURED_REPAIR] model_service不可用，跳过自修复: context_type=patch_path_plan")
                return None

            repair_prompt = (
                "你上一次输出的JSON结构有误，请修复后重新输出。\n\n"
                "【错误信息】JSON解析失败\n\n"
                f"【你的原始输出】\n{raw_output}\n\n"
                '【期望格式】\n{"dimensions": {"维度名": ["路径1", "路径2"], ...}}\n\n'
                "【原始请求（完整上下文）】\n"
                f"{original_prompt}\n\n"
                "请直接输出修复后的JSON，不要输出其他内容。 /no_think"
            )

            messages = [
                {"role": "system", "content": _PATCH_SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ]

            logger.info("[STRUCTURED_REPAIR] 尝试自修复: context_type=patch_path_plan")
            repair_response = model_service.call_model(messages)

            if not repair_response:
                return None

            return self._parse_result(repair_response, vacancy_dimensions)

        except Exception as e:
            logger.warning(
                f"[STRUCTURED_REPAIR] 自修复异常: error_type={type(e).__name__}, "
                f"context_type=patch_path_plan"
            )
            return None
