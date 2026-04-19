# -*- coding: utf-8 -*-
"""
回答生成Chain策略

实现基于知识素材的回答生成Chain策略。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult
from src.orchestration.model_business_service.Impl.consult_model_service import ConsultModelService

logger = logging.getLogger(__name__)

DISCLAIMER = "以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"

# 回答长度控制常量
MIN_WORDS = 200
MAX_WORDS = 800


@dataclass
class AnswerGenerationContextBody:
    """
    回答生成Chain策略专属输入数据类

    Attributes:
        query_text: 用户查询文本
        knowledge_context: 整合后的知识素材文本
        intent_label: 意图标签
        chat_history: 对话历史
    """
    query_text: str
    knowledge_context: str = ""
    intent_label: str = ""
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query_text": self.query_text,
            "knowledge_context": self.knowledge_context,
            "intent_label": self.intent_label,
            "chat_history": self.chat_history
        }


@dataclass
class AnswerGenerationResultData:
    """
    回答生成Chain策略专属输出数据类

    Attributes:
        answer_text: 生成的回答文本
        sources: 知识来源引用列表
        word_count: 回答字数
        has_disclaimer: 是否包含免责声明
    """
    answer_text: str
    sources: List[str] = field(default_factory=list)
    word_count: int = 0
    has_disclaimer: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "answer_text": self.answer_text,
            "sources": self.sources,
            "word_count": self.word_count,
            "has_disclaimer": self.has_disclaimer
        }


@dataclass
class AnswerGenerationResource:
    """
    回答生成Chain策略专属资源类

    Attributes:
        model_service: 咨询模型服务
    """
    model_service: Optional[ConsultModelService] = None

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


class AnswerGenerationChain(Chain[ChainContext[AnswerGenerationContextBody], ChainResult[AnswerGenerationResultData]]):
    """
    回答生成Chain策略类

    实现基于知识素材的回答生成固定流程：
    1. 构建提示词
    2. 调用模型生成回答
    3. 格式化回答（添加免责声明、标记来源）
    4. 质量检查
    """

    def __init__(self, resource: AnswerGenerationResource):
        """
        初始化回答生成Chain策略

        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource

    def execute(self, chain_context: ChainContext[AnswerGenerationContextBody]) -> ChainResult[AnswerGenerationResultData]:
        """
        执行Chain策略

        Args:
            chain_context: Chain输入数据容器

        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[AnswerGenerationChain] 开始执行Chain: session_id={chain_context.session_id}")

        body = chain_context.body
        if body is None:
            logger.warning(f"[AnswerGenerationChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=AnswerGenerationResultData(answer_text="输入数据为空", has_disclaimer=False)
            )

        logger.info(f"[AnswerGenerationChain] 开始构建提示词: query_text={body.query_text[:50]}...")
        prompt = self._build_prompt(body)

        messages = self._truncate_chat_history(body.chat_history) + [
            {"role": "system", "content": prompt["system_message"]},
            {"role": "user", "content": prompt["user_message"]}
        ]

        logger.info(f"[AnswerGenerationChain] 开始模型推理: messages_count={len(messages)}")
        try:
            raw_answer = self._resource.get_model_result(messages)
        except Exception as e:
            logger.error(f"[AnswerGenerationChain] 模型推理异常: {e}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=AnswerGenerationResultData(answer_text=f"模型推理失败: {str(e)}", has_disclaimer=False)
            )
        logger.info(f"[AnswerGenerationChain] 模型推理完成: answer_len={len(raw_answer)}")

        sources = self._extract_sources(body.knowledge_context)
        formatted_answer = self._format_answer(raw_answer, sources)
        logger.info(f"[AnswerGenerationChain] 回答格式化完成: formatted_len={len(formatted_answer)}")
        
        # 长度检查和调整
        adjusted_answer = self._check_and_adjust_length(formatted_answer, body)
        logger.info(f"[AnswerGenerationChain] 长度调整完成: adjusted_len={len(adjusted_answer)}")

        quality_passed = self._check_quality(adjusted_answer)
        logger.info(f"[AnswerGenerationChain] 质量检查结果: passed={quality_passed}")

        result_data = AnswerGenerationResultData(
            answer_text=adjusted_answer,
            sources=sources,
            word_count=len(adjusted_answer),
            has_disclaimer=DISCLAIMER in adjusted_answer
        )

        elapsed = time.time() - start_time
        logger.info(f"[AnswerGenerationChain] Chain执行完成: session_id={chain_context.session_id}, "
                    f"word_count={result_data.word_count}, has_disclaimer={result_data.has_disclaimer}, "
                    f"quality_passed={quality_passed}, elapsed={elapsed:.2f}s")

        return ChainResult(session_id=chain_context.session_id, data=result_data)

    async def execute_stream(self, chain_context) -> AsyncGenerator[str, None]:
        context_body = chain_context.body
        if context_body is None:
            yield "抱歉，无法处理您的咨询请求。"
            return
        
        prompt = self._build_prompt(context_body)
        
        if self._resource is None or self._resource.model_service is None:
            yield "抱歉，模型服务不可用。"
            return
        
        model_service = self._resource.model_service
        if hasattr(model_service, 'get_model_result'):
            model_service = model_service.get_model_result()
        
        if model_service is None:
            yield "抱歉，模型服务不可用。"
            return
        
        try:
            full_response = []
            truncated_history = self._truncate_chat_history(context_body.chat_history)
            messages = truncated_history + [
                {"role": "system", "content": prompt["system_message"]},
                {"role": "user", "content": prompt["user_message"]}
            ]
            async for token in model_service.async_stream_generate(messages):
                full_response.append(token)
                yield token
            
            complete_answer = ''.join(full_response)
            
            if "以上信息仅供参考" not in complete_answer and "不构成医疗建议" not in complete_answer:
                disclaimer = "\n\n以上信息仅供参考，不构成医疗建议。如有健康问题，请及时就医。"
                for char in disclaimer:
                    full_response.append(char)
                    yield char
                complete_answer += disclaimer
            
            logger.info(f"[AnswerGenerationChain] ========== LLM完整输出 ==========")
            logger.info(f"[AnswerGenerationChain] 完整回答 (长度={len(complete_answer)}):")
            logger.info(f"{complete_answer}")
            logger.info(f"[AnswerGenerationChain] ==============================")
                
        except Exception as e:
            logger.error(f"[AnswerGenerationChain] 流式生成异常: {str(e)}")
            yield f"\n\n抱歉，生成过程中出现错误。"

    def _truncate_chat_history(self, chat_history: List[Dict[str, str]], max_rounds: int = 2, max_assistant_len: int = 200) -> List[Dict[str, str]]:
        if not chat_history:
            return []

        recent = chat_history[-(max_rounds * 2):]

        truncated = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant" and len(content) > max_assistant_len:
                content = content[:max_assistant_len] + "..."
            truncated.append({"role": role, "content": content})

        return truncated

    MAX_KNOWLEDGE_CHARS = 6000

    def _build_prompt(self, context_body: AnswerGenerationContextBody) -> Dict[str, str]:
        system_message = """你是一位专业的医疗健康咨询助手，请根据提供的医学知识素材回答用户的问题。

重要要求：
1. 仅针对用户问题回答，不要对回答本身进行评价或总结
2. 不要添加"回答内容特点"、"以上是回答"等自我评价内容
3. 不要在回答末尾添加"✅"等标记或总结性文字"""

        if context_body.knowledge_context:
            knowledge = context_body.knowledge_context
            if len(knowledge) > self.MAX_KNOWLEDGE_CHARS:
                knowledge = knowledge[:self.MAX_KNOWLEDGE_CHARS] + "\n...(知识素材已截断)"
            system_message += f"\n\n参考知识素材：\n{knowledge}"

        user_message = f"用户问题：{context_body.query_text}"

        user_message += (
            "\n\n回答要求："
            "\n1.回答长度200-800字"
            "\n2.基于知识素材回答，不要编造信息"
            "\n3.如果知识素材不足以回答问题，请如实说明"
            "\n4.仅针对用户问题回答，不要添加自我评价或总结"
            f"\n\n在回答末尾必须添加免责声明：'{DISCLAIMER}'"
        )

        return {
            "system_message": system_message,
            "user_message": user_message
        }

    def _format_answer(self, raw_answer: str, sources: List[str]) -> str:
        """
        格式化回答

        Args:
            raw_answer: 原始回答
            sources: 知识来源列表

        Returns:
            格式化后的回答
        """
        formatted = raw_answer

        if DISCLAIMER not in formatted:
            formatted = formatted.rstrip() + "\n\n" + DISCLAIMER

        if sources:
            source_text = "参考来源：" + "、".join(sources)
            formatted = formatted.rstrip() + "\n" + source_text

        return formatted

    def _check_quality(self, answer: str) -> bool:
        """
        质量检查

        Args:
            answer: 回答文本

        Returns:
            质量检查是否通过
        """
        answer_len = len(answer)
        if answer_len < 200 or answer_len > 800:
            logger.warning(f"[AnswerGenerationChain] 质量检查未通过: 回答长度={answer_len}, 要求200-800字")
            return False

        if DISCLAIMER not in answer:
            logger.warning("[AnswerGenerationChain] 质量检查未通过: 缺少免责声明")
            return False

        return True

    def _extract_sources(self, knowledge_context: str) -> List[str]:
        """
        从知识上下文中提取来源

        Args:
            knowledge_context: 知识上下文

        Returns:
            来源列表
        """
        sources = []
        if not knowledge_context:
            return sources

        for line in knowledge_context.split("\n"):
            line = line.strip()
            if line.startswith("疾病名称：") or line.startswith("疾病:"):
                source_name = line.split("：", 1)[-1].strip() if "：" in line else line.split(":", 1)[-1].strip()
                if source_name:
                    sources.append(source_name)

        return sources
    
    def _check_and_adjust_length(self, answer: str, context_body: AnswerGenerationContextBody) -> str:
        """
        检查并调整回答长度
        
        Args:
            answer: 原始回答
            context_body: 回答生成专属输入数据
            
        Returns:
            调整后的回答
        """
        answer_len = len(answer)
        logger.info(f"[AnswerGenerationChain] 检查回答长度: current_len={answer_len}, min={MIN_WORDS}, max={MAX_WORDS}")
        
        if answer_len < MIN_WORDS:
            logger.warning(f"[AnswerGenerationChain] 回答过短: {answer_len} < {MIN_WORDS}, 开始扩展")
            return self._expand_answer(answer, context_body)
        elif answer_len > MAX_WORDS:
            logger.warning(f"[AnswerGenerationChain] 回答过长: {answer_len} > {MAX_WORDS}, 开始精简")
            return self._compress_answer(answer)
        else:
            logger.info(f"[AnswerGenerationChain] 回答长度符合要求: {answer_len}")
            return answer
    
    def _expand_answer(self, answer: str, context_body: AnswerGenerationContextBody) -> str:
        """
        扩展不足200字的回答
        
        Args:
            answer: 原始回答
            context_body: 回答生成专属输入数据
            
        Returns:
            扩展后的回答
        """
        # 移除免责声明，稍后重新添加
        answer_without_disclaimer = answer.replace(DISCLAIMER, "").strip()
        
        # 构建扩展提示
        expand_prompt = f"""
当前回答过于简短，请根据以下知识素材扩展回答，使其更加详细和完整：

用户问题：{context_body.query_text}

知识素材：
{context_body.knowledge_context}

当前回答：
{answer_without_disclaimer}

扩展要求：
1. 基于知识素材扩展，不要编造信息
2. 增加相关细节和解释
3. 保持回答的专业性和准确性
4. 扩展后的回答长度应在200-800字之间
"""
        
        messages = [
            {"role": "system", "content": "你是一位专业的医疗健康咨询助手，请根据提供的医学知识素材扩展回答。"},
            {"role": "user", "content": expand_prompt}
        ]
        
        try:
            expanded_answer = self._resource.get_model_result(messages)
            # 确保包含免责声明
            if DISCLAIMER not in expanded_answer:
                expanded_answer = expanded_answer.rstrip() + "\n\n" + DISCLAIMER
            logger.info(f"[AnswerGenerationChain] 回答扩展完成: original_len={len(answer)}, expanded_len={len(expanded_answer)}")
            return expanded_answer
        except Exception as e:
            logger.error(f"[AnswerGenerationChain] 回答扩展失败: {e}")
            # 扩展失败时返回原回答
            return answer
    
    def _compress_answer(self, answer: str) -> str:
        """
        精简超过800字的回答
        
        Args:
            answer: 原始回答
            
        Returns:
            精简后的回答
        """
        # 移除免责声明，稍后重新添加
        answer_without_disclaimer = answer.replace(DISCLAIMER, "").strip()
        
        # 构建精简提示
        compress_prompt = f"""
当前回答过长，请精简以下回答，保留核心信息：

当前回答：
{answer_without_disclaimer}

精简要求：
1. 保留核心信息和关键内容
2. 删除冗余和重复内容
3. 保持回答的准确性和专业性
4. 精简后的回答长度应在200-800字之间
"""
        
        messages = [
            {"role": "system", "content": "你是一位专业的医疗健康咨询助手，请精简回答，保留核心信息。"},
            {"role": "user", "content": compress_prompt}
        ]
        
        try:
            compressed_answer = self._resource.get_model_result(messages)
            # 确保包含免责声明
            if DISCLAIMER not in compressed_answer:
                compressed_answer = compressed_answer.rstrip() + "\n\n" + DISCLAIMER
            logger.info(f"[AnswerGenerationChain] 回答精简完成: original_len={len(answer)}, compressed_len={len(compressed_answer)}")
            return compressed_answer
        except Exception as e:
            logger.error(f"[AnswerGenerationChain] 回答精简失败: {e}")
            # 精简失败时返回原回答
            return answer
