# -*- coding: utf-8 -*-
"""
报告生成Chain策略

实现健康报告生成业务的报告生成Chain策略。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult

logger = logging.getLogger(__name__)

# 免责声明
DISCLAIMER = "以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"

# 报告长度控制常量
MIN_WORDS = 1000
MAX_WORDS = 5000

# 最大重试次数
MAX_RETRY_COUNT = 2


@dataclass
class ReportGenerationContextBody:
    """
    报告生成Chain策略专属输入数据类

    Attributes:
        report_materials: 报告素材
        health_score: 健康评分
        health_level: 健康等级
        risk_level: 风险等级
        risk_diseases: 风险疾病
        user_profile: 用户档案
        monitoring_data: 监测数据
    """
    report_materials: Dict = field(default_factory=dict)
    health_score: int = 0
    health_level: str = ""
    risk_level: str = ""
    risk_diseases: List[Dict] = field(default_factory=list)
    user_profile: Dict = field(default_factory=dict)
    monitoring_data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_materials": self.report_materials,
            "health_score": self.health_score,
            "health_level": self.health_level,
            "risk_level": self.risk_level,
            "risk_diseases": self.risk_diseases,
            "user_profile": self.user_profile,
            "monitoring_data": self.monitoring_data
        }


@dataclass
class ReportGenerationResultData:
    """
    报告生成Chain策略专属输出数据类

    Attributes:
        report_content: 报告内容（Markdown格式）
        word_count: 报告字数
        has_disclaimer: 是否包含免责声明
        sources: 知识来源
    """
    report_content: str = ""
    word_count: int = 0
    has_disclaimer: bool = False
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_content": self.report_content,
            "word_count": self.word_count,
            "has_disclaimer": self.has_disclaimer,
            "sources": self.sources
        }


@dataclass
class ReportGenerationResource:
    """
    报告生成Chain策略专属资源类

    Attributes:
        model_service: 报告模型服务（将在后续实现ReportModelService）
    """
    model_service: Optional[Any] = None

    def get_model_result(self, messages: List[Dict[str, str]]) -> str:
        """
        获取模型生成结果

        Args:
            messages: 消息列表

        Returns:
            模型生成的回复
        """
        if self.model_service is None:
            return "模型服务未初始化"
        return self.model_service.call_model(messages)


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
        logger.info(f"[ReportGenerationChain] 开始执行Chain: session_id={chain_context.session_id}")

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

            messages = [
                {"role": "system", "content": prompt["system_message"]},
                {"role": "user", "content": prompt["user_message"]}
            ]

            # 调用LLM生成报告
            logger.info(f"[ReportGenerationChain] 开始调用LLM: messages_count={len(messages)}")
            try:
                report_content = self._resource.get_model_result(messages)
                logger.info(f"[ReportGenerationChain] LLM调用完成: report_len={len(report_content)}")
            except Exception as e:
                logger.error(f"[ReportGenerationChain] LLM调用异常: {e}")
                # 启用降级策略
                self._is_degraded = True
                report_content = self._generate_degraded_report(body)
                logger.info(f"[ReportGenerationChain] 使用降级策略生成报告: report_len={len(report_content)}")
                break

            # 内容校验
            quality_passed = self._check_quality(report_content, body)
            logger.info(f"[ReportGenerationChain] 内容校验结果: passed={quality_passed}")

            if quality_passed:
                break

            retry_count += 1
            if retry_count <= MAX_RETRY_COUNT:
                logger.warning(f"[ReportGenerationChain] 报告不符合要求，准备重试: retry_count={retry_count}")

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

    def execute_stream(self, chain_context: ChainContext[ReportGenerationContextBody]) -> Generator[str, None, None]:
        """
        流式执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Yields:
            生成的报告内容片段
        """
        context_body = chain_context.body
        if context_body is None:
            yield "抱歉，无法生成健康报告。"
            return

        # 构建提示词
        prompt = self._build_prompt(context_body)

        # 检查模型服务是否可用
        if self._resource is None or self._resource.model_service is None:
            logger.warning("[ReportGenerationChain] 模型服务不可用，使用降级策略")
            self._is_degraded = True
            degraded_report = self._generate_degraded_report(context_body)
            for char in degraded_report:
                yield char
            return

        model_service = self._resource.model_service

        # 检查是否支持流式生成
        if not hasattr(model_service, 'stream_generate_with_context'):
            logger.warning("[ReportGenerationChain] 模型服务不支持流式生成，使用普通生成")
            messages = [
                {"role": "system", "content": prompt["system_message"]},
                {"role": "user", "content": prompt["user_message"]}
            ]
            try:
                report_content = self._resource.get_model_result(messages)
                for char in report_content:
                    yield char
                # 添加免责声明
                if DISCLAIMER not in report_content:
                    for char in "\n\n" + DISCLAIMER:
                        yield char
            except Exception as e:
                logger.error(f"[ReportGenerationChain] 模型生成失败: {str(e)}")
                self._is_degraded = True
                degraded_report = self._generate_degraded_report(context_body)
                for char in degraded_report:
                    yield char
            return

        # 使用流式生成
        try:
            full_response = []

            # 构建流式生成的上下文
            knowledge_context = self._build_knowledge_context(context_body.report_materials)

            for token in model_service.stream_generate_with_context(
                user_query=prompt["user_message"],
                knowledge_context=knowledge_context
            ):
                full_response.append(token)
                yield token

            # 添加免责声明
            disclaimer = "\n\n" + DISCLAIMER
            for char in disclaimer:
                full_response.append(char)
                yield char

            # 记录完整的LLM输出
            complete_report = ''.join(full_response)
            logger.info(f"[ReportGenerationChain] ========== LLM完整输出 ==========")
            logger.info(f"[ReportGenerationChain] 完整报告 (长度={len(complete_report)}):")
            logger.info(f"{complete_report}")
            logger.info(f"[ReportGenerationChain] ==============================")

        except Exception as e:
            logger.error(f"[ReportGenerationChain] 流式生成异常: {str(e)}")
            yield f"\n\n抱歉，报告生成过程中出现错误。"

    def _build_prompt(self, context_body: ReportGenerationContextBody) -> Dict[str, str]:
        """
        构建提示词

        Args:
            context_body: 报告生成专属输入数据

        Returns:
            包含system_message和user_message的字典
        """
        # 系统指令
        system_message = "你是一位专业的医疗健康评估助手，请根据提供的监测数据和医学知识生成健康评估报告。"

        # 添加知识素材
        knowledge_context = self._build_knowledge_context(context_body.report_materials)
        if knowledge_context:
            system_message += f"\n\n参考知识素材：\n{knowledge_context}"

        # 构建用户消息（包含报告结构模板和用户数据）
        user_message = self._build_user_message(context_body)

        return {
            "system_message": system_message,
            "user_message": user_message
        }

    def _build_knowledge_context(self, report_materials: Dict) -> str:
        """
        构建知识素材上下文

        Args:
            report_materials: 报告素材

        Returns:
            知识素材文本
        """
        if not report_materials:
            return ""

        context_parts = []

        merged_results = report_materials.get("merged_results", [])
        if merged_results:
            context_parts.append("=== 知识检索结果 ===")
            for item in merged_results:
                if isinstance(item, dict):
                    entity = item.get("entity", "")
                    data = item.get("data", {})
                    if entity and data:
                        context_parts.append(f"实体：{entity}")
                        if isinstance(data, dict):
                            for key, value in data.items():
                                if value:
                                    context_parts.append(f"  {key}：{value}")
                        else:
                            context_parts.append(f"  数据：{data}")

        dimension_results = report_materials.get("dimension_results", {})
        if dimension_results:
            context_parts.append("\n=== 8维度评估结果 ===")
            dimension_names = {
                "dimension_1": "疾病风险评估",
                "dimension_2": "用药建议",
                "dimension_3": "治疗方案",
                "dimension_4": "饮食建议",
                "dimension_5": "检查建议",
                "dimension_6": "并发症预警",
                "dimension_7": "预防措施",
                "dimension_8": "易感人群"
            }
            for dim_key, dim_result in dimension_results.items():
                dim_name = dimension_names.get(dim_key, dim_key)
                if isinstance(dim_result, dict):
                    score = dim_result.get("score", dim_result.get("confidence", "未知"))
                    level = dim_result.get("level", "未知")
                    analysis = dim_result.get("analysis", dim_result.get("evaluation_result", ""))
                    context_parts.append(f"\n【{dim_name}】")
                    if isinstance(score, (int, float)):
                        context_parts.append(f"  置信度：{score:.2f}")
                    else:
                        context_parts.append(f"  置信度：{score}")
                    if analysis and isinstance(analysis, dict):
                        for key, value in analysis.items():
                            if value:
                                context_parts.append(f"  {key}：{value}")
                    elif analysis:
                        context_parts.append(f"  详情：{analysis}")

        anomalies = report_materials.get("anomalies", [])
        if anomalies:
            context_parts.append("\n=== 异常指标 ===")
            for anomaly in anomalies:
                if isinstance(anomaly, dict):
                    indicator = anomaly.get("indicator", anomaly.get("name", "未知指标"))
                    value = anomaly.get("value", "")
                    reference = anomaly.get("reference", "")
                    severity = anomaly.get("severity", "")
                    context_parts.append(f"- {indicator}：{value}")
                    if reference:
                        context_parts.append(f"  参考范围：{reference}")
                    if severity:
                        context_parts.append(f"  严重程度：{severity}")

        risk_factors = report_materials.get("risk_factors", [])
        if risk_factors:
            context_parts.append("\n=== 风险因素 ===")
            for factor in risk_factors:
                if isinstance(factor, dict):
                    name = factor.get("factor_name", factor.get("name", factor.get("factor", "未知因素")))
                    risk_level = factor.get("risk_level", "")
                    basis = factor.get("basis", "")
                    context_parts.append(f"- {name}")
                    if risk_level:
                        context_parts.append(f"  风险等级：{risk_level}")
                    if basis:
                        context_parts.append(f"  依据：{basis}")

        return "\n".join(context_parts)

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
                from datetime import datetime
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            except (ValueError, TypeError):
                pass

        gender = user_profile.get('gender', '未知') if user_profile else '未知'
        height = user_profile.get('height', '未知') if user_profile else '未知'
        weight = user_profile.get('weight', '未知') if user_profile else '未知'

        # 病史字段（字符串类型）
        past_medical_history = user_profile.get('past_medical_history', '') if user_profile else ''
        family_history = user_profile.get('family_history', '') if user_profile else ''
        allergy_history = user_profile.get('allergy_history', '') if user_profile else ''
        surgical_history = user_profile.get('surgical_history', '') if user_profile else ''
        medical_compliance = user_profile.get('medical_compliance', '') if user_profile else ''

        user_info = f"""
用户基本信息：
- 年龄：{age}岁
- 性别：{gender}
- 身高：{height} cm
- 体重：{weight} kg
- 既往病史：{past_medical_history if past_medical_history else '无'}
- 家族病史：{family_history if family_history else '无'}
- 过敏史：{allergy_history if allergy_history else '无'}
- 手术史：{surgical_history if surgical_history else '无'}
- 用药医嘱：{medical_compliance if medical_compliance else '无'}
"""

        monitoring_data_text = ""
        report_materials = context_body.report_materials or {}
        monitoring_data = context_body.monitoring_data or report_materials.get("monitoring_data", {})
        if monitoring_data:
            monitoring_parts = ["\n监测数据："]

            # 心率数据
            heart_rate = monitoring_data.get('heart_rate', {})
            if isinstance(heart_rate, dict) and heart_rate.get('latest'):
                latest_hr_list = heart_rate['latest']
                if isinstance(latest_hr_list, list) and latest_hr_list:
                    latest_hr = latest_hr_list[-1] if isinstance(latest_hr_list[-1], dict) else {}
                    hr_value = latest_hr.get('value', '未知')
                    monitoring_parts.append(f"- 心率：{hr_value} 次/分钟")

            # 血压数据
            blood_pressure = monitoring_data.get('blood_pressure', {})
            if isinstance(blood_pressure, dict) and blood_pressure.get('latest'):
                latest_bp_list = blood_pressure['latest']
                if isinstance(latest_bp_list, list) and latest_bp_list:
                    latest_bp = latest_bp_list[-1] if isinstance(latest_bp_list[-1], dict) else {}
                    systolic = latest_bp.get('systolic', '未知')
                    diastolic = latest_bp.get('diastolic', '未知')
                    monitoring_parts.append(f"- 血压：收缩压 {systolic} mmHg，舒张压 {diastolic} mmHg")

            # 血糖数据
            blood_glucose = monitoring_data.get('blood_glucose', {})
            if isinstance(blood_glucose, dict) and blood_glucose.get('latest'):
                latest_glucose_list = blood_glucose['latest']
                if isinstance(latest_glucose_list, list) and latest_glucose_list:
                    latest_glucose = latest_glucose_list[-1] if isinstance(latest_glucose_list[-1], dict) else {}
                    glucose_value = latest_glucose.get('value', '未知')
                    glucose_type = latest_glucose.get('type', '空腹')
                    monitoring_parts.append(f"- 血糖：{glucose_type}血糖 {glucose_value} mmol/L")

            # 血氧数据
            blood_oxygen = monitoring_data.get('blood_oxygen', {})
            if isinstance(blood_oxygen, dict) and blood_oxygen.get('latest'):
                latest_oxygen_list = blood_oxygen['latest']
                if isinstance(latest_oxygen_list, list) and latest_oxygen_list:
                    latest_oxygen = latest_oxygen_list[-1] if isinstance(latest_oxygen_list[-1], dict) else {}
                    oxygen_value = latest_oxygen.get('value', '未知')
                    monitoring_parts.append(f"- 血氧：{oxygen_value}%")

            # 睡眠数据
            sleep = monitoring_data.get('sleep', {})
            if isinstance(sleep, dict) and sleep.get('latest'):
                latest_sleep_list = sleep['latest']
                if isinstance(latest_sleep_list, list) and latest_sleep_list:
                    latest_sleep = latest_sleep_list[-1] if isinstance(latest_sleep_list[-1], dict) else {}
                    sleep_value = latest_sleep.get('value', '未知')
                    monitoring_parts.append(f"- 睡眠：{sleep_value} 小时")

            # 灌注指数数据
            perfusion_index = monitoring_data.get('perfusion_index', {})
            if isinstance(perfusion_index, dict) and perfusion_index.get('latest'):
                latest_pi_list = perfusion_index['latest']
                if isinstance(latest_pi_list, list) and latest_pi_list:
                    latest_pi = latest_pi_list[-1] if isinstance(latest_pi_list[-1], dict) else {}
                    pi_value = latest_pi.get('value', '未知')
                    monitoring_parts.append(f"- 灌注指数：{pi_value} PI")

            if len(monitoring_parts) > 1:
                monitoring_data_text = "\n".join(monitoring_parts)

        risk_diseases_text = ""
        if context_body.risk_diseases:
            disease_names = [d.get("name", d.get("disease_name", "")) for d in context_body.risk_diseases if d.get("name") or d.get("disease_name")]
            if disease_names:
                risk_diseases_text = "、".join(disease_names)

        report_template = f"""
请根据以上信息生成一份完整的健康评估报告，报告结构如下：

# 健康评估报告

## 一、健康综合评分
[{context_body.health_score}分]（{context_body.health_level}）

## 二、监测数据分析
[异常指标分析]

## 三、风险评估
[{context_body.risk_level}]：{risk_diseases_text}

## 四、各维度评估
[8个维度的评估结果]

## 五、健康建议
[综合建议]

## 六、免责声明
以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。
"""

        user_message = f"""
{user_info}
{monitoring_data_text}

{report_template}

生成要求：
1. 报告长度：{MIN_WORDS}-{MAX_WORDS}字
2. 必须包含免责声明
3. 必须包含健康评分
4. 基于提供的知识素材进行分析，不要编造信息
5. 语言专业、准确、易懂
6. 在报告末尾必须添加免责声明：'{DISCLAIMER}'
"""

        return user_message

    def _check_quality(self, report_content: str, context_body: ReportGenerationContextBody) -> bool:
        """
        内容校验（长度、格式、免责声明）

        Args:
            report_content: 报告内容
            context_body: 报告生成专属输入数据

        Returns:
            质量检查是否通过
        """
        # 长度检查
        report_len = len(report_content)
        if report_len < MIN_WORDS or report_len > MAX_WORDS:
            logger.warning(f"[ReportGenerationChain] 质量检查未通过: 报告长度={report_len}, 要求{MIN_WORDS}-{MAX_WORDS}字")
            return False

        # 免责声明检查
        if DISCLAIMER not in report_content:
            logger.warning("[ReportGenerationChain] 质量检查未通过: 缺少免责声明")
            return False

        # 健康评分检查
        if str(context_body.health_score) not in report_content:
            logger.warning(f"[ReportGenerationChain] 质量检查未通过: 缺少健康评分{context_body.health_score}")
            return False

        logger.info(f"[ReportGenerationChain] 质量检查通过: report_len={report_len}")
        return True

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

        # 用户信息
        user_profile = context_body.user_profile

        # 计算年龄
        age = "未知"
        birth_date = user_profile.get('birth_date') if user_profile else None
        if birth_date:
            try:
                from datetime import datetime
                birth = datetime.strptime(birth_date, "%Y-%m-%d")
                today = datetime.now()
                age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
            except (ValueError, TypeError):
                pass

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
