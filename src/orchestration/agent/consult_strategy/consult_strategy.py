# -*- coding: utf-8 -*-
"""
健康咨询Agent策略

实现健康咨询Agent策略。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.orchestration.agent.agent_strategy import AgentStrategy
from src.orchestration.agent.data_classes import AgentContext, AgentResult
from src.orchestration.agent.agent_resource import AgentResource

logger = logging.getLogger(__name__)


@dataclass
class ConsultContextBody:
    """
    健康咨询Agent策略专属输入数据类
    
    Attributes:
        question: 用户问题
        session_id: 会话ID
        conversation_history: 对话历史
        user_profile: 用户档案
    """
    question: str
    session_id: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "question": self.question,
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "user_profile": self.user_profile
        }


@dataclass
class ConsultResultData:
    """
    健康咨询Agent策略专属输出数据类
    
    Attributes:
        answer: 咨询答案
        suggestions: 健康建议
        related_knowledge: 相关知识
        follow_up_questions: 后续问题
        confidence: 置信度
    """
    answer: str
    suggestions: List[str] = field(default_factory=list)
    related_knowledge: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "answer": self.answer,
            "suggestions": self.suggestions,
            "related_knowledge": self.related_knowledge,
            "follow_up_questions": self.follow_up_questions,
            "confidence": self.confidence
        }


class ConsultStrategy(AgentStrategy[ConsultContextBody, ConsultResultData]):
    """
    健康咨询Agent策略类
    
    实现健康咨询的Agent策略，使用状态机管理流程：
    1. INIT: 初始化状态
    2. RETRIEVE_KNOWLEDGE: 检索知识
    3. GENERATE_ANSWER: 生成答案
    4. COMPLETE: 完成状态
    """
    
    def execute(
        self,
        context: AgentContext[ConsultContextBody],
        resource: AgentResource
    ) -> AgentResult[ConsultResultData]:
        """
        执行Agent策略
        
        Args:
            context: Agent输入数据容器
            resource: Agent资源
            
        Returns:
            AgentResult: Agent输出数据容器
        """
        start_time = time.time()
        logger.info(f"[ConsultStrategy] 开始执行策略: session_id={context.session_id}")
        
        body = context.body
        if body is None:
            logger.warning(f"[ConsultStrategy] 输入数据为空: session_id={context.session_id}")
            return AgentResult(
                session_id=context.session_id,
                data=ConsultResultData(answer="输入数据为空", confidence=0.0)
            )
        
        logger.info(f"[ConsultStrategy] 问题: {body.question[:100]}...")
        
        knowledge_chain = resource.get_chain("knowledge_chain")
        if knowledge_chain is None:
            logger.error(f"[ConsultStrategy] 知识检索链未初始化: session_id={context.session_id}")
            return AgentResult(
                session_id=context.session_id,
                data=ConsultResultData(answer="知识检索链未初始化", confidence=0.0)
            )
        
        from src.orchestration.chain.data_classes import ChainContext
        from src.orchestration.chain.consult_with_knowledge_chain import ConsultWithKnowledgeContextBody
        
        chain_context = ChainContext(
            session_id=context.session_id,
            body=ConsultWithKnowledgeContextBody(
                question=body.question,
                user_id=body.user_profile.get("user_id", "") if body.user_profile else "",
                conversation_history=body.conversation_history or []
            )
        )
        
        logger.info(f"[ConsultStrategy] 调用知识检索链: session_id={context.session_id}")
        chain_result = knowledge_chain.execute(chain_context)
        
        if chain_result.data is None:
            logger.warning(f"[ConsultStrategy] 知识检索失败: session_id={context.session_id}")
            return AgentResult(
                session_id=context.session_id,
                data=ConsultResultData(answer="知识检索失败", confidence=0.0)
            )
        
        result_data = ConsultResultData(
            answer=chain_result.data.answer,
            related_knowledge=chain_result.data.knowledge_used,
            confidence=chain_result.data.confidence
        )
        
        suggestions = self._generate_suggestions(body.question, chain_result.data.knowledge_used)
        result_data.suggestions = suggestions
        
        follow_up = self._generate_follow_up_questions(body.question)
        result_data.follow_up_questions = follow_up
        
        elapsed = time.time() - start_time
        logger.info(f"[ConsultStrategy] 策略执行完成: session_id={context.session_id}, "
                   f"confidence={result_data.confidence}, elapsed={elapsed:.2f}s")
        
        return AgentResult(session_id=context.session_id, data=result_data)
    
    def _generate_suggestions(self, question: str, knowledge: List[str]) -> List[str]:
        """
        生成健康建议
        
        Args:
            question: 用户问题
            knowledge: 相关知识
            
        Returns:
            健康建议列表
        """
        suggestions = []
        
        if "头痛" in question or "头晕" in question:
            suggestions.append("建议保持充足睡眠，避免过度劳累")
            suggestions.append("如症状持续，建议及时就医检查")
        
        if knowledge:
            suggestions.append("建议了解更多相关知识，做好预防措施")
        
        if not suggestions:
            suggestions.append("建议保持健康的生活方式")
            suggestions.append("如有不适，请及时就医")
        
        return suggestions
    
    def _generate_follow_up_questions(self, question: str) -> List[str]:
        """
        生成后续问题
        
        Args:
            question: 用户问题
            
        Returns:
            后续问题列表
        """
        follow_up = []
        
        if "症状" in question or "不舒服" in question:
            follow_up.append("您的症状持续多长时间了？")
            follow_up.append("是否有其他伴随症状？")
        
        if "药物" in question or "治疗" in question:
            follow_up.append("您是否对某些药物过敏？")
            follow_up.append("您目前是否正在服用其他药物？")
        
        if not follow_up:
            follow_up.append("您还有其他健康问题想咨询吗？")
        
        return follow_up[:3]
