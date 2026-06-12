# -*- coding: utf-8 -*-
"""
报告生成Chain策略

实现健康报告生成业务的报告生成Chain策略。
"""

import logging
import statistics
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, List
from src.config.business.report_service_config import get_runtime_config
from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.chain.report_generation_chain.report_generation_context import ReportGenerationContextBody
from src.orchestration.chain.report_generation_chain.report_generation_result import ReportGenerationResultData
from src.orchestration.chain.report_generation_chain.report_generation_resource import ReportGenerationResource
from src.utils.logger import log_arch_event

logger = logging.getLogger(__name__)

# 报告业务配置（延迟加载代理，确保运行期获取YAML配置值）
class _LazyReportConfig:
    """延迟加载配置代理，每次属性访问时从ConfigManager获取真实配置"""
    def __getattr__(self, name):
        return getattr(get_runtime_config(), name)

_report_config = _LazyReportConfig()

# 免责声明
DISCLAIMER = "以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"

# 报告长度控制常量（从配置读取）
MIN_WORDS = _report_config.report_min_length
MAX_WORDS = _report_config.report_max_length

# Prompt长度控制常量（避免超过模型上下文长度）
MAX_PROMPT_CHARS = _report_config.max_prompt_chars
MAX_KNOWLEDGE_CHARS = _report_config.max_knowledge_chars

# 最大重试次数
MAX_RETRY_COUNT = _report_config.max_report_retries


def _refresh_report_config():
    """刷新report_generation_chain模块级常量，从ConfigManager获取正确配置值。"""
    global MIN_WORDS, MAX_WORDS, MAX_PROMPT_CHARS, MAX_KNOWLEDGE_CHARS, MAX_RETRY_COUNT
    MIN_WORDS = _report_config.report_min_length
    MAX_WORDS = _report_config.report_max_length
    MAX_PROMPT_CHARS = _report_config.max_prompt_chars
    MAX_KNOWLEDGE_CHARS = _report_config.max_knowledge_chars
    MAX_RETRY_COUNT = _report_config.max_report_retries

class ReportGenerationChain(Chain[ChainContext[ReportGenerationContextBody], ChainResult[ReportGenerationResultData]]):
    """
    报告生成Chain策略类

    实现健康报告生成的固定流程：
    1. 构建提示词（系统指令+报告结构+知识素材+用户数据）
    2. 调用LLM生成报告
    3. 内容校验（长度、格式、免责声明）
    4. 重试机制（最多2次）
    """

    def __init__(self, resource: ReportGenerationResource):
        """
        初始化报告生成Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource
        self._is_degraded = False

    def execute(self, chain_context: ChainContext[ReportGenerationContextBody]) -> ChainResult[ReportGenerationResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        _refresh_report_config()
        logger.info(f"[ReportGenerationChain] 开始执行Chain: session_id={chain_context.session_id}")
        log_arch_event(
            logger,
            component="ReportGenerationChain",
            stage="CHAIN",
            event="execute",
            status="start",
            design_id="BIZ-4.4",
        )

        body = chain_context.body
        if body is None:
            logger.warning(f"[ReportGenerationChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=ReportGenerationResultData(
                    report_content="输入数据为空",
                    has_disclaimer=False
                )
            )

        # 重置降级状态
        self._is_degraded = False

        # 尝试生成报告（最多重试MAX_RETRY_COUNT次）
        retry_count = 0
        report_content = ""
        quality_passed = False

        while retry_count <= MAX_RETRY_COUNT:
            logger.info(f"[ReportGenerationChain] 开始生成报告: retry_count={retry_count}")

            # 构建提示词
            logger.info(f"[ReportGenerationChain] 开始构建提示词: health_score={body.health_score}")
            prompt = self._build_prompt(body)

            # [TOKEN_BUDGET] 日志
            prompt_total_chars = len(prompt["system_message"]) + len(prompt["user_message"])
            logger.info(f"[ReportGenerationChain] [TOKEN_BUDGET] prompt_total_chars={prompt_total_chars}, MAX_PROMPT_CHARS={MAX_PROMPT_CHARS}, estimated_tokens={prompt_total_chars//4}")

            messages = [
                {"role": "system", "content": prompt["system_message"]},
                {"role": "user", "content": prompt["user_message"]}
            ]

            # 调用LLM生成报告
            logger.info(f"[ReportGenerationChain] 开始调用LLM: messages_count={len(messages)}")
            prompt_total_chars = sum(len(msg.get('content', '')) for msg in messages)
            logger.info(f"[ReportGenerationChain] [LLM_INPUT] messages_count={len(messages)}, total_chars={prompt_total_chars}")
            for i, msg in enumerate(messages):
                logger.info(f"[ReportGenerationChain] [LLM_INPUT] Message[{i}] role={msg.get('role')}: {msg.get('content', '')}")
            llm_start_time = time.time()
            try:
                report_content = self._resource.get_model_result(
                    messages,
                    temperature=_report_config.report_generation_temperature,
                    max_tokens=_report_config.report_generation_max_tokens
                )
                llm_duration = time.time() - llm_start_time
                logger.info(f"[ReportGenerationChain] [LLM_OUTPUT] report_len={len(report_content)}")
                logger.info(f"[ReportGenerationChain] [LLM_OUTPUT] {report_content}")
                logger.info(f"[ReportGenerationChain] [LLM_DURATION] duration={llm_duration:.2f}s")
                logger.info(f"[ReportGenerationChain] LLM调用完成: report_len={len(report_content)}, duration={llm_duration:.2f}s")
            except Exception as e:
                logger.error(f"[ReportGenerationChain] LLM调用异常: {e}")
                llm_duration = time.time() - llm_start_time
                logger.info(f"[ReportGenerationChain] [LLM_DURATION] duration={llm_duration:.2f}s (failed)")
                # 启用降级策略
                self._is_degraded = True
                report_content = self._generate_degraded_report(body)
                logger.info(f"[ReportGenerationChain] [LLM_DEGRADED] reason=LLM调用异常({e}), report_len={len(report_content)}")
                logger.info(f"[ReportGenerationChain] 使用降级策略生成报告: report_len={len(report_content)}")
                break

            # 内容校验
            quality_passed, quality_fail_reason = self._check_quality(report_content, body)
            logger.info(f"[ReportGenerationChain] 内容校验结果: passed={quality_passed}")

            if quality_passed:
                break

            retry_count += 1
            if retry_count <= MAX_RETRY_COUNT:
                logger.warning(f"[ReportGenerationChain] 报告不符合要求，准备重试: retry_count={retry_count}, fail_reason={quality_fail_reason}")

        # 提取知识来源
        sources = self._extract_sources(body.report_materials)

        # 确保报告包含免责声明
        if DISCLAIMER not in report_content:
            report_content = report_content.rstrip() + "\n\n" + DISCLAIMER

        result_data = ReportGenerationResultData(
            report_content=report_content,
            word_count=len(report_content),
            has_disclaimer=DISCLAIMER in report_content,
            sources=sources
        )

        elapsed = time.time() - start_time
        logger.info(f"[ReportGenerationChain] Chain执行完成: session_id={chain_context.session_id}, "
                    f"word_count={result_data.word_count}, has_disclaimer={result_data.has_disclaimer}, "
                    f"quality_passed={quality_passed}, is_degraded={self._is_degraded}, "
                    f"retry_count={retry_count}, elapsed={elapsed:.2f}s")

        return ChainResult(session_id=chain_context.session_id, data=result_data)

    async def execute_stream(self, chain_context) -> AsyncGenerator[str, None]:
        _refresh_report_config()
        context_body = chain_context.body
        if context_body is None:
            yield "抱歉，无法生成健康报告。"
            return

        prompt = self._build_prompt(context_body)
        
        if self._resource is None or self._resource.model_service is None:
            yield "抱歉，模型服务不可用。"
            return

        model_service = self._resource.model_service

        if model_service is None:
            yield "抱歉，模型服务不可用。"
            return

        try:
            full_response = []

            messages = [
                {"role": "system", "content": prompt["system_message"]},
                {"role": "user", "content": prompt["user_message"]}
            ]

            prompt_total_chars = sum(len(msg.get('content', '')) for msg in messages)
            logger.info(f"[ReportGenerationChain] [LLM_INPUT] messages_count={len(messages)}, total_chars={prompt_total_chars}")
            for i, msg in enumerate(messages):
                logger.info(f"[ReportGenerationChain] [LLM_INPUT] Message[{i}] role={msg.get('role')}: {msg.get('content', '')}")

            llm_start_time = time.time()
            async for token in model_service.async_stream_generate(messages):
                full_response.append(token)
                yield token

            llm_duration = time.time() - llm_start_time

            complete_report = ''.join(full_response)

            if DISCLAIMER not in complete_report:
                disclaimer = "\n\n" + DISCLAIMER
                for char in disclaimer:
                    full_response.append(char)
                    yield char
                complete_report += disclaimer

            logger.info(f"[ReportGenerationChain] [LLM_OUTPUT] report_len={len(complete_report)}")
            logger.info(f"[ReportGenerationChain] [LLM_OUTPUT] {complete_report}")
            logger.info(f"[ReportGenerationChain] [LLM_DURATION] duration={llm_duration:.2f}s")

        except Exception as e:
            logger.error(f"[ReportGenerationChain] 流式生成异常: {str(e)}")
            self._is_degraded = True
            logger.info(f"[ReportGenerationChain] [LLM_DEGRADED] reason=流式生成异常({str(e)})")
            logger.warning("[ReportGenerationChain] 流式生成失败，降级为规则引擎生成报告")
            try:
                degraded_content = self._generate_degraded_report(context_body)
                logger.info(f"[ReportGenerationChain] [LLM_DEGRADED] degraded_report_len={len(degraded_content)}")
                yield degraded_content
            except Exception as de:
                logger.error(f"[ReportGenerationChain] 降级报告生成也失败: {de}")
                yield "\n\n抱歉，报告生成过程中出现错误。"

    def _build_prompt(self, context_body: ReportGenerationContextBody) -> Dict[str, str]:
        """
        构建提示词

        Args:
            context_body: 报告生成专属输入数据

        Returns:
            包含system_message和user_message的字典
        """
        # 系统指令
        system_message = """你是一位专业的医疗健康评估助手。请严格按照用户提供的报告模板结构生成健康评估报告。

输出要求：
1. 直接以"# 健康评估报告"开头，以免责声明结束
2. 严格按模板输出六个章节，不得增减
3. 第一至第四节：专业学术分析风格，语言严谨准确，不使用图标、特殊符号、emoji表情，仅进行数据分析，不提出建议
4. 第五节：针对老年用户，采用适老化表达，直接、明确地提出建议

禁止输出：
- 禁止输出"分析部分"、"建议部分"等分类标题
- 禁止输出"全文完"、"报告完毕"、"总字数："等结束语或统计信息
- 禁止输出任何报告正文之外的说明、总结、提示等内容
- 禁止输出对本次生成任务的完成汇报
- 禁止对报告内容进行自我评价或总结（如"✅ 报告特点"、"以上是报告内容"等）

只输出报告正文内容，不输出任何附加内容。"""

        # 添加知识素材
        knowledge_context = self._build_knowledge_context(context_body.report_materials)
        if knowledge_context:
            system_message += f"\n\n参考知识素材：\n{knowledge_context}"

        # 构建用户消息（包含报告结构模板和用户数据）
        user_message = self._build_user_message(context_body)

        # [PROMPT_CONSTRUCTION] 日志：方法入口
        logger.info(f"[ReportGenerationChain] [PROMPT_CONSTRUCTION] 开始构建Prompt: system_message_len={len(system_message)}, user_data_len={len(user_message)}")

        prompt = {
            "system_message": system_message,
            "user_message": user_message
        }

        # [PROMPT_CONSTRUCTION] 日志：方法返回
        knowledge_len = len(knowledge_context) if knowledge_context else 0
        total_len = len(system_message) + len(user_message)
        logger.info(f"[ReportGenerationChain] [PROMPT_CONSTRUCTION] Prompt构建完成: total_len={total_len}, knowledge_len={knowledge_len}")

        return prompt

    def _build_knowledge_context(self, report_materials: Dict) -> str:
        """
        构建知识素材上下文
        
        添加长度限制，避免超过模型上下文长度（8192 tokens）
        
        Args:
            report_materials: 报告素材
        
        Returns:
            知识素材文本
        """
        if not report_materials:
            return ""
        
        context_parts = []
        total_chars = 0
        
        merged_results = report_materials.get("merged_results", [])
        if merged_results:
            context_parts.append("=== 知识检索结果 ===")
            total_chars += len(context_parts[-1])
            
            for item in merged_results:
                if total_chars >= MAX_KNOWLEDGE_CHARS:
                    logger.warning("[ReportGenerationChain] 知识素材长度达到上限，截断merged_results")
                    logger.warning(f"[ReportGenerationChain] [KNOWLEDGE_TRUNCATION] 截断知识素材: key=merged_results, original_len={total_chars}, truncated_len={MAX_KNOWLEDGE_CHARS}")
                    break
                
                if isinstance(item, dict):
                    entity = item.get("entity", "")
                    data = item.get("data", {})
                    if entity and data:
                        entity_text = f"实体：{entity}"
                        context_parts.append(entity_text)
                        total_chars += len(entity_text)
                        
                        if isinstance(data, dict):
                            for key, value in data.items():
                                if value:
                                    if total_chars >= MAX_KNOWLEDGE_CHARS:
                                        break
                                    value_text = f"  {key}：{value}"
                                    if len(value_text) > _report_config.value_text_max_chars:
                                        value_text = value_text[:_report_config.value_text_max_chars] + "..."
                                    context_parts.append(value_text)
                                    total_chars += len(value_text)
                        else:
                            data_text = f"  数据：{data}"
                            context_parts.append(data_text)
                            total_chars += len(data_text)
        
        if total_chars >= MAX_KNOWLEDGE_CHARS:
            return "\n".join(context_parts)
        
        dimension_results = report_materials.get("dimension_results", {})
        if dimension_results:
            # Check if dimension_summaries has _degraded marker
            is_degraded = any(
                dim_data.get('_degraded', False)
                for dim_data in dimension_results.values()
                if isinstance(dim_data, dict)
            )
            if is_degraded:
                partial_note = "\n注意：以下知识数据因超时降级可能不完整（部分数据），仅供参考。"
                context_parts.append(partial_note)
                total_chars += len(partial_note)

            dim_header = "\n=== 8维度知识检索 ==="
            context_parts.append(dim_header)
            total_chars += len(dim_header)
            
            dimension_names = {
                "disease_risk": "疾病风险评估",
                "medication": "用药建议",
                "treatment": "治疗方案",
                "dietary": "饮食建议",
                "checkup": "检查建议",
                "complication": "并发症预警",
                "prevention": "预防措施",
                "susceptible": "易感人群"
            }
            
            for dim_key, dim_result in dimension_results.items():
                if total_chars >= MAX_KNOWLEDGE_CHARS:
                    logger.warning("[ReportGenerationChain] 知识素材长度达到上限，截断dimension_results")
                    logger.warning(f"[ReportGenerationChain] [KNOWLEDGE_TRUNCATION] 截断知识素材: key=dimension_results, original_len={total_chars}, truncated_len={MAX_KNOWLEDGE_CHARS}")
                    break
                
                dim_name = dimension_names.get(dim_key, dim_key)
                if isinstance(dim_result, dict):
                    summary = dim_result.get("summary", "")

                    dim_text = f"\n【{dim_name}】"
                    context_parts.append(dim_text)
                    total_chars += len(dim_text)
                    
                    if summary:
                        summary_text = f"  摘要：{summary}"
                        if len(summary_text) > _report_config.summary_text_max_chars:
                            summary_text = summary_text[:_report_config.summary_text_max_chars] + "..."
                        context_parts.append(summary_text)
                        total_chars += len(summary_text)
        
        if total_chars >= MAX_KNOWLEDGE_CHARS:
            return "\n".join(context_parts)
        
        anomalies = report_materials.get("anomalies", [])
        if anomalies:
            anomaly_header = "\n=== 异常指标 ==="
            context_parts.append(anomaly_header)
            total_chars += len(anomaly_header)
            
            for anomaly in anomalies:
                if total_chars >= MAX_KNOWLEDGE_CHARS:
                    logger.warning("[ReportGenerationChain] 知识素材长度达到上限，截断anomalies")
                    logger.warning(f"[ReportGenerationChain] [KNOWLEDGE_TRUNCATION] 截断知识素材: key=anomalies, original_len={total_chars}, truncated_len={MAX_KNOWLEDGE_CHARS}")
                    break
                
                if isinstance(anomaly, dict):
                    indicator = anomaly.get("indicator", anomaly.get("indicator_name", anomaly.get("name", "未知指标")))
                    anomaly_type = anomaly.get("anomaly_type", "")
                    anomaly_value = anomaly.get("anomaly_value", anomaly.get("value", ""))
                    reference = anomaly.get("reference", anomaly.get("reference_range", ""))
                    severity = anomaly.get("severity", "")

                    parts = [p for p in [indicator, anomaly_type, anomaly_value] if p]
                    indicator_text = f"- {'：'.join(parts)}" if len(parts) > 1 else f"- {parts[0]}" if parts else ""
                    if indicator_text:
                        context_parts.append(indicator_text)
                        total_chars += len(indicator_text)
                    
                    if reference:
                        ref_text = f"  参考范围：{reference}"
                        context_parts.append(ref_text)
                        total_chars += len(ref_text)
                    
                    if severity:
                        sev_text = f"  严重程度：{severity}"
                        context_parts.append(sev_text)
                        total_chars += len(sev_text)
        
        if total_chars >= MAX_KNOWLEDGE_CHARS:
            return "\n".join(context_parts)
        
        risk_factors = report_materials.get("risk_factors", [])
        if risk_factors:
            risk_header = "\n=== 风险因素 ==="
            context_parts.append(risk_header)
            total_chars += len(risk_header)
            
            for factor in risk_factors:
                if total_chars >= MAX_KNOWLEDGE_CHARS:
                    logger.warning("[ReportGenerationChain] 知识素材长度达到上限，截断risk_factors")
                    logger.warning(f"[ReportGenerationChain] [KNOWLEDGE_TRUNCATION] 截断知识素材: key=risk_factors, original_len={total_chars}, truncated_len={MAX_KNOWLEDGE_CHARS}")
                    break
                
                if isinstance(factor, dict):
                    name = factor.get("factor_name", factor.get("name", factor.get("factor", "未知因素")))
                    risk_level = factor.get("risk_level", "")
                    basis = factor.get("basis", "")
                    
                    name_text = f"- {name}"
                    context_parts.append(name_text)
                    total_chars += len(name_text)
                    
                    if risk_level:
                        level_text = f"  风险等级：{risk_level}"
                        context_parts.append(level_text)
                        total_chars += len(level_text)
                    
                    if basis:
                        basis_text = f"  依据：{basis}"
                        if len(basis_text) > _report_config.basis_text_max_chars:
                            basis_text = basis_text[:_report_config.basis_text_max_chars] + "..."
                        context_parts.append(basis_text)
                        total_chars += len(basis_text)
        
        result = "\n".join(context_parts)
        logger.info(f"[ReportGenerationChain] 知识素材构建完成: chars={len(result)}, truncated={len(result) >= MAX_KNOWLEDGE_CHARS}")
        
        return result

    def _build_user_message(self, context_body: ReportGenerationContextBody) -> str:
        """
        构建用户消息（包含报告结构模板和用户数据）

        使用新的UserProfile字段：
        - user_id, gender, birth_date, height, weight
        - past_medical_history, family_history, allergy_history, surgical_history, medical_compliance

        Args:
            context_body: 报告生成专属输入数据

        Returns:
            用户消息文本
        """
        user_profile = context_body.user_profile

        # 计算年龄
        age = "未知"
        birth_date = user_profile.get('birth_date') if user_profile else None
        if birth_date:
            try:
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            except (ValueError, TypeError):
                profile_age = user_profile.get('age') if user_profile else None
                age = str(int(profile_age)) if isinstance(profile_age, (int, float)) and profile_age > 0 else "未知"
        elif user_profile:
            profile_age = user_profile.get('age')
            age = str(int(profile_age)) if isinstance(profile_age, (int, float)) and profile_age > 0 else "未知"

        gender = user_profile.get('gender', '未知') or '未知' if user_profile else '未知'
        height = user_profile.get('height') or '未知' if user_profile else '未知'
        weight = user_profile.get('weight') or '未知' if user_profile else '未知'

        # 病史字段（字符串类型）
        past_medical_history = user_profile.get('past_medical_history', '') or '' if user_profile else ''
        family_history = user_profile.get('family_history', '') or '' if user_profile else ''
        allergy_history = user_profile.get('allergy_history', '') or '' if user_profile else ''
        surgical_history = user_profile.get('surgical_history', '') or '' if user_profile else ''
        medical_compliance = user_profile.get('medical_compliance', '') or '' if user_profile else ''

        # 构建用户信息，跳过未提供的数值型字段
        user_info_lines = [
            f"- 年龄：{age}岁",
            f"- 性别：{gender}",
        ]
        if height != '未知':
            user_info_lines.append(f"- 身高：{height} cm")
        if weight != '未知':
            user_info_lines.append(f"- 体重：{weight} kg")
        user_info_lines.extend([
            f"- 既往病史：{past_medical_history if past_medical_history else '无'}",
            f"- 家族病史：{family_history if family_history else '无'}",
            f"- 过敏史：{allergy_history if allergy_history else '无'}",
            f"- 手术史：{surgical_history if surgical_history else '无'}",
            f"- 用药医嘱：{medical_compliance if medical_compliance else '无'}",
        ])

        user_info = "用户基本信息：\n" + "\n".join(user_info_lines) + "\n"

        monitoring_data_text = ""
        report_materials = context_body.report_materials or {}
        monitoring_data = context_body.monitoring_data or report_materials.get("monitoring_data", {})
        if monitoring_data:
            monitoring_parts = ["\n监测数据（阶段性统计特征）："]

            indicators = [
                ('heart_rate', '心率', '次/分钟', 'value'),
                ('blood_glucose', '血糖', 'mmol/L', 'value'),
                ('perfusion_index', '灌注指数', 'PI', 'value'),
                ('blood_oxygen', '血氧', '%', 'value'),
                ('sleep', '睡眠', '小时', 'duration'),
            ]

            for indicator_key, indicator_name, unit, value_key in indicators:
                indicator_data = monitoring_data.get(indicator_key, {})
                if isinstance(indicator_data, dict):
                    stats_text = self._extract_indicator_stats(indicator_data, indicator_name, unit, value_key)
                    if stats_text:
                        monitoring_parts.append(stats_text)

            blood_pressure = monitoring_data.get('blood_pressure', {})
            if isinstance(blood_pressure, dict):
                bp_stats_text = self._extract_blood_pressure_stats(blood_pressure)
                if bp_stats_text:
                    monitoring_parts.append(bp_stats_text)

            if len(monitoring_parts) > 1:
                monitoring_data_text = "\n".join(monitoring_parts)

        risk_diseases_text = ""
        if context_body.risk_diseases:
            disease_names = [d.get("name", d.get("disease_name", "")) for d in context_body.risk_diseases if d.get("name") or d.get("disease_name")]
            if disease_names:
                risk_diseases_text = "、".join(disease_names)

        report_template = f"""
【报告结构模板 - 必须严格按此结构输出，不得输出任何模板说明文字】

# 健康评估报告

---

## 一、健康综合评分

**评分：{context_body.health_score:.2f}分（{context_body.health_level}）**

[详细说明评分依据、评分等级含义、与用户健康状况的对应关系]

---

## 二、监测数据分析

[根据提供的阶段性监测数据统计特征，逐项详细分析各指标：
 - 分析各项指标的当前数值、变化趋势、波动情况
 - 结合正常医学参考范围进行对比分析
 - 分析指标之间的关联性和相互影响
 - 识别异常指标并分析其可能的医学意义
 - 每项指标分析应包含数据解读和临床意义说明]

---

## 三、风险评估

**风险等级：{context_body.risk_level}**

**风险疾病：{risk_diseases_text if risk_diseases_text else '暂无'}**

[详细分析风险因素：
 - 逐一分析各风险因素的来源、严重程度
 - 结合用户病史、家族史、监测数据进行综合评估
 - 说明各风险因素之间的关联性
 - 分析潜在的健康威胁和发展趋势]

---

## 四、各维度评估

基于提供的用户信息、监测数据、风险因素及参考评估结果，对用户进行以下维度的综合评估：

### 1. 整体健康状态评估

[综合所有生理指标和风险因素，评估用户当前整体健康状态]

### 2. 慢性病管理效果评估

[基于血压、血糖等慢性病相关指标的变化趋势，评估慢性病控制效果]

### 3. 生活方式健康度评估

[基于睡眠、作息等生活方式相关指标，评估生活方式健康程度]

### 4. 疾病发展趋势评估

[基于当前指标变化趋势和风险因素，评估潜在疾病发展趋势]

### 5. 健康风险预警评估

[综合风险因素和异常指标，评估需要重点警惕的健康隐患]

### 6. 健康改善空间评估

[分析当前健康状况与理想状态的差距，评估可优化的健康方向]

---

## 五、健康建议

[针对用户具体情况，提出详细、具体、可操作的健康建议：
 - 每条建议应明确说明具体做法
 - 建议内容应便于老年用户理解和执行
 - 建议应覆盖日常生活的各个方面]

---

## 六、免责声明

> {DISCLAIMER}
"""

        user_message = f"""
{user_info}
{monitoring_data_text}

{report_template}

【输出规范 - 必须严格遵守】

1. 报告长度：{MIN_WORDS}-{MAX_WORDS}字，内容要详实丰富

2. 内容要求：
   - 第一至第四节（健康评分、监测数据分析、风险评估、各维度评估）：采用专业学术分析风格，语言严谨准确，不使用任何图标、特殊符号、emoji表情，仅进行数据分析，不提出建议。每个章节内容要详实，分析要深入透彻。
   - 第五节（健康建议）：针对老年用户，采用适老化表达，直接、明确地提出建议。建议要具体详细，便于执行。

3. 禁止输出的内容：
   - 禁止输出"分析部分"、"建议部分"等分类标题
   - 禁止输出"全文完"、"报告完毕"、"总字数："等结束语或统计信息
   - 禁止输出任何报告正文之外的说明、总结、提示等内容
   - 禁止输出对本次生成任务的完成汇报

4. 格式要求：
   - 直接以"# 健康评估报告"开头
   - 严格按模板结构输出六个章节
   - 以免责声明结束，之后不得有任何内容

5. 内容丰富度要求：
   - 监测数据分析章节：每项指标分析不少于100字
   - 风险评估章节：每个风险因素分析不少于50字
   - 各维度评估章节：6个评估维度，每个维度评估不少于80字
   - 健康建议章节：建议条目不少于5条，每条不少于30字
"""

        return user_message

    def _extract_indicator_stats(self, indicator_data: Dict, indicator_name: str, unit: str, value_key: str) -> str:
        """
        提取单个监测指标的统计特征

        Args:
            indicator_data: 指标数据字典，包含latest, daily_stats, weekly_stats, monthly_stats
            indicator_name: 指标名称
            unit: 单位
            value_key: 值字段名称

        Returns:
            格式化的统计特征文本
        """

        stats_parts = [f"\n【{indicator_name}】"]

        latest_data = indicator_data.get('latest', [])
        if latest_data and isinstance(latest_data, list):
            values = []
            for item in latest_data:
                if isinstance(item, dict):
                    val = item.get(value_key)
                    if val is not None and isinstance(val, (int, float)):
                        values.append(float(val))
            if values:
                stats_parts.append(f"  最新值: {values[-1]:.1f} {unit}")
                if len(values) > 1:
                    stats_parts.append(f"  近期均值（共{len(values)}次）: {statistics.mean(values):.1f} {unit}")
                    stats_parts.append(f"  近期波动: {min(values):.1f} - {max(values):.1f} {unit}")

        daily_stats = indicator_data.get('daily_stats', [])
        if daily_stats and isinstance(daily_stats, list):
            daily_values = []
            for item in daily_stats:
                if isinstance(item, dict):
                    val = item.get('avg_value')
                    if val is not None and isinstance(val, (int, float)):
                        daily_values.append(float(val))
            if daily_values:
                count = len(daily_values)
                stats_parts.append(f"  日均值（共{count}天）: {statistics.mean(daily_values):.1f} {unit}")
                stats_parts.append(f"  日最高值: {max(daily_values):.1f} {unit}")
                stats_parts.append(f"  日最低值: {min(daily_values):.1f} {unit}")
                if count > 1:
                    stats_parts.append(f"  日标准差: {statistics.stdev(daily_values):.2f} {unit}")

        weekly_stats = indicator_data.get('weekly_stats', [])
        if weekly_stats and isinstance(weekly_stats, list):
            weekly_values = []
            for item in weekly_stats:
                if isinstance(item, dict):
                    val = item.get('avg_value')
                    if val is not None and isinstance(val, (int, float)):
                        weekly_values.append(float(val))
            if weekly_values:
                count = len(weekly_values)
                stats_parts.append(f"  周均值（共{count}周）: {statistics.mean(weekly_values):.1f} {unit}")
                if count >= 2:
                    recent_half = weekly_values[:max(1, count//2)]
                    older_half = weekly_values[max(1, count//2):max(1, count//2)*2] if count > 2 else weekly_values
                    if recent_half and older_half:
                        trend = "上升" if statistics.mean(recent_half) > statistics.mean(older_half) else "下降" if statistics.mean(recent_half) < statistics.mean(older_half) else "稳定"
                        stats_parts.append(f"  趋势: {trend}")

        monthly_stats = indicator_data.get('monthly_stats', [])
        if monthly_stats and isinstance(monthly_stats, list):
            monthly_values = []
            for item in monthly_stats:
                if isinstance(item, dict):
                    val = item.get('avg_value')
                    if val is not None and isinstance(val, (int, float)):
                        monthly_values.append(float(val))
            if monthly_values:
                count = len(monthly_values)
                stats_parts.append(f"  月均值（共{count}月）: {statistics.mean(monthly_values):.1f} {unit}")

        return "\n".join(stats_parts) if len(stats_parts) > 1 else ""

    def _extract_blood_pressure_stats(self, bp_data: Dict) -> str:
        """
        提取血压的特殊统计特征（收缩压和舒张压）

        Args:
            bp_data: 血压数据字典

        Returns:
            格式化的血压统计特征文本
        """

        stats_parts = ["\n【血压】"]

        latest_data = bp_data.get('latest', [])
        if latest_data and isinstance(latest_data, list):
            systolic_values = []
            diastolic_values = []
            for item in latest_data:
                if isinstance(item, dict):
                    sys = item.get('systolic')
                    dia = item.get('diastolic')
                    if sys is not None and isinstance(sys, (int, float)):
                        systolic_values.append(float(sys))
                    if dia is not None and isinstance(dia, (int, float)):
                        diastolic_values.append(float(dia))
            if systolic_values and diastolic_values:
                stats_parts.append(f"  最新值: {systolic_values[-1]:.0f}/{diastolic_values[-1]:.0f} mmHg")
                if len(systolic_values) > 1:
                    stats_parts.append(f"  近期均值（共{len(systolic_values)}次）: {statistics.mean(systolic_values):.0f}/{statistics.mean(diastolic_values):.0f} mmHg")

        daily_stats = bp_data.get('daily_stats', [])
        if daily_stats and isinstance(daily_stats, list):
            daily_sys = []
            daily_dia = []
            for item in daily_stats:
                if isinstance(item, dict):
                    sys = item.get('avg_systolic')
                    dia = item.get('avg_diastolic')
                    if sys is not None and isinstance(sys, (int, float)):
                        daily_sys.append(float(sys))
                    if dia is not None and isinstance(dia, (int, float)):
                        daily_dia.append(float(dia))
            if daily_sys and daily_dia:
                count = len(daily_sys)
                stats_parts.append(f"  日均值（共{count}天）: {statistics.mean(daily_sys):.0f}/{statistics.mean(daily_dia):.0f} mmHg")
                stats_parts.append(f"  日最高值: {max(daily_sys):.0f}/{max(daily_dia):.0f} mmHg")
                stats_parts.append(f"  日最低值: {min(daily_sys):.0f}/{min(daily_dia):.0f} mmHg")
                if count > 1:
                    stats_parts.append(f"  日标准差: 收缩压{statistics.stdev(daily_sys):.1f}/舒张压{statistics.stdev(daily_dia):.1f} mmHg")

        weekly_stats = bp_data.get('weekly_stats', [])
        if weekly_stats and isinstance(weekly_stats, list):
            weekly_sys = []
            weekly_dia = []
            for item in weekly_stats:
                if isinstance(item, dict):
                    sys = item.get('avg_systolic')
                    dia = item.get('avg_diastolic')
                    if sys is not None and isinstance(sys, (int, float)):
                        weekly_sys.append(float(sys))
                    if dia is not None and isinstance(dia, (int, float)):
                        weekly_dia.append(float(dia))
            if weekly_sys and weekly_dia:
                count = len(weekly_sys)
                stats_parts.append(f"  周均值（共{count}周）: {statistics.mean(weekly_sys):.0f}/{statistics.mean(weekly_dia):.0f} mmHg")

        monthly_stats = bp_data.get('monthly_stats', [])
        if monthly_stats and isinstance(monthly_stats, list):
            monthly_sys = []
            monthly_dia = []
            for item in monthly_stats:
                if isinstance(item, dict):
                    sys = item.get('avg_systolic')
                    dia = item.get('avg_diastolic')
                    if sys is not None and isinstance(sys, (int, float)):
                        monthly_sys.append(float(sys))
                    if dia is not None and isinstance(dia, (int, float)):
                        monthly_dia.append(float(dia))
            if monthly_sys and monthly_dia:
                count = len(monthly_sys)
                stats_parts.append(f"  月均值（共{count}月）: {statistics.mean(monthly_sys):.0f}/{statistics.mean(monthly_dia):.0f} mmHg")

        return "\n".join(stats_parts) if len(stats_parts) > 1 else ""

    def _check_quality(self, report_content: str, context_body: ReportGenerationContextBody) -> tuple:
        """
        内容校验（长度、格式、免责声明）

        Args:
            report_content: 报告内容
            context_body: 报告生成专属输入数据

        Returns:
            (质量检查是否通过, 失败原因)
        """
        # 长度检查
        report_len = len(report_content)
        if report_len < MIN_WORDS:
            return False, f"报告长度={report_len}, 低于最低要求{MIN_WORDS}字"
        if report_len > MAX_WORDS:
            return False, f"报告长度={report_len}, 超过最高限制{MAX_WORDS}字"

        # 免责声明检查
        if DISCLAIMER not in report_content:
            return False, "缺少免责声明"

        # 健康评分检查
        if f"{context_body.health_score:.2f}" not in report_content:
            return False, f"缺少健康评分{context_body.health_score:.2f}"

        logger.info(f"[ReportGenerationChain] 质量检查通过: report_len={report_len}")
        return True, ""

    def _extract_sources(self, report_materials: Dict) -> List[str]:
        """
        从报告素材中提取知识来源

        Args:
            report_materials: 报告素材

        Returns:
            来源列表
        """
        sources = []
        if not report_materials:
            return sources

        merged_results = report_materials.get("merged_results", [])
        if not merged_results:
            return sources

        for item in merged_results:
            if isinstance(item, dict):
                entity = item.get("entity", "")
                if entity and entity not in sources:
                    sources.append(entity)

        return sources

    def _generate_degraded_report(self, context_body: ReportGenerationContextBody) -> str:
        """
        降级策略：生成简化报告

        当LLM不可用时，使用预设模板生成简化报告

        Args:
            context_body: 报告生成专属输入数据

        Returns:
            简化报告内容
        """
        logger.info("[ReportGenerationChain] 使用降级策略生成简化报告")
        logger.info(f"[ReportGenerationChain] [LLM_DEGRADED] method=_generate_degraded_report, health_score={context_body.health_score}, risk_level={context_body.risk_level}")

        # 用户信息
        user_profile = context_body.user_profile

        # 计算年龄
        age = "未知"
        birth_date = user_profile.get('birth_date') if user_profile else None
        if birth_date:
            try:
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            except (ValueError, TypeError):
                profile_age = user_profile.get('age') if user_profile else None
                age = str(int(profile_age)) if isinstance(profile_age, (int, float)) and profile_age > 0 else "未知"
        elif user_profile:
            profile_age = user_profile.get('age')
            age = str(int(profile_age)) if isinstance(profile_age, (int, float)) and profile_age > 0 else "未知"

        gender = user_profile.get('gender', '未知') if user_profile else '未知'

        user_info = ""
        if user_profile:
            user_info = f"""
**用户信息**
- 年龄：{age}
- 性别：{gender}
- 既往病史：{user_profile.get('past_medical_history', '无')}
- 家族病史：{user_profile.get('family_history', '无')}
"""

        # 风险疾病
        risk_diseases_text = "暂无"
        if context_body.risk_diseases:
            disease_names = [d.get("disease_name", d.get("name", "")) for d in context_body.risk_diseases if d.get("disease_name") or d.get("name")]
            if disease_names:
                risk_diseases_text = "、".join(disease_names)

        # 简化报告模板
        report = f"""# 健康评估报告

## 一、健康综合评分
**{context_body.health_score}分**（{context_body.health_level}）

{user_info}

## 二、风险评估
**风险等级**：{context_body.risk_level}

**风险疾病**：{risk_diseases_text}

## 三、健康建议
根据您的健康评分和风险评估结果，建议：
1. 定期进行健康体检
2. 保持良好的生活习惯
3. 如有不适症状，及时就医

## 四、免责声明
{DISCLAIMER}
"""

        return report
