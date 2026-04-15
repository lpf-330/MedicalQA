# -*- coding: utf-8 -*-
"""
健康咨询Chain策略

实现带知识检索的健康咨询Chain策略。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.orchestration.chain.chain import Chain
from src.orchestration.chain.data_classes import ChainContext, ChainResult

logger = logging.getLogger(__name__)


@dataclass
class ConsultWithKnowledgeContextBody:
    """
    健康咨询Chain策略专属输入数据类
    
    Attributes:
        question: 用户问题
        user_id: 用户ID
        conversation_history: 对话历史
    """
    question: str
    user_id: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "question": self.question,
            "user_id": self.user_id,
            "conversation_history": self.conversation_history
        }


@dataclass
class ConsultWithKnowledgeResultData:
    """
    健康咨询Chain策略专属输出数据类
    
    Attributes:
        answer: 咨询答案
        knowledge_used: 使用的知识来源
        confidence: 置信度
    """
    answer: str
    knowledge_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "answer": self.answer,
            "knowledge_used": self.knowledge_used,
            "confidence": self.confidence
        }


@dataclass
class ConsultWithKnowledgeResource:
    """
    健康咨询Chain策略专属资源类
    
    Attributes:
        neo4j_handler: Neo4j医疗知识图谱调用处理器
        model_service: 模型业务服务
    """
    neo4j_handler: Optional[Any] = None
    model_service: Optional[Any] = None
    
    def get_knowledge(self, query: str) -> Dict[str, Any]:
        """
        获取相关知识
        
        Args:
            query: 查询内容
            
        Returns:
            知识结果
        """
        if self.neo4j_handler is None:
            return {}
        
        result = self.neo4j_handler.get_disease_info(query)
        if result:
            symptoms = self.neo4j_handler.get_symptoms_by_disease(query)
            drugs = self.neo4j_handler.get_drugs_by_disease(query)
            foods = self.neo4j_handler.get_foods_by_disease(query)
            
            return {
                "disease_info": result,
                "symptoms": symptoms,
                "drugs": drugs,
                "foods": foods
            }
        
        diseases = self.neo4j_handler.search_diseases_by_symptom(query)
        return {"possible_diseases": diseases}
    
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


class ConsultWithKnowledgeChain(Chain[ConsultWithKnowledgeContextBody, ConsultWithKnowledgeResultData]):
    """
    健康咨询Chain策略类
    
    实现带知识检索的健康咨询固定流程：
    1. 根据用户问题检索相关知识
    2. 将知识上下文和用户问题组合
    3. 调用模型生成回答
    """
    
    def __init__(self, resource: ConsultWithKnowledgeResource):
        """
        初始化健康咨询Chain策略
        
        Args:
            resource: Chain策略专属资源
        """
        self._resource = resource
    
    def execute(self, chain_context: ChainContext[ConsultWithKnowledgeContextBody]) -> ChainResult[ConsultWithKnowledgeResultData]:
        """
        执行Chain策略
        
        Args:
            chain_context: Chain输入数据容器
            
        Returns:
            ChainResult: Chain输出数据容器
        """
        start_time = time.time()
        logger.info(f"[ConsultWithKnowledgeChain] 开始执行Chain: session_id={chain_context.session_id}")
        
        body = chain_context.body
        if body is None:
            logger.warning(f"[ConsultWithKnowledgeChain] 输入数据为空: session_id={chain_context.session_id}")
            return ChainResult(
                session_id=chain_context.session_id,
                data=ConsultWithKnowledgeResultData(answer="输入数据为空", confidence=0.0)
            )
        
        logger.info(f"[ConsultWithKnowledgeChain] 开始知识检索: question={body.question[:50]}...")
        knowledge = self._resource.get_knowledge(body.question)
        logger.info(f"[ConsultWithKnowledgeChain] 知识检索完成: found={bool(knowledge)}")
        
        knowledge_context = self._build_knowledge_context(knowledge)
        logger.debug(f"[ConsultWithKnowledgeChain] 知识上下文: {knowledge_context[:200]}...")
        
        messages = body.conversation_history + [
            {"role": "system", "content": f"参考知识：\n{knowledge_context}"},
            {"role": "user", "content": body.question}
        ]
        
        logger.info(f"[ConsultWithKnowledgeChain] 开始模型推理: messages_count={len(messages)}")
        answer = self._resource.get_model_result(messages)
        logger.info(f"[ConsultWithKnowledgeChain] 模型推理完成: answer_len={len(answer)}")
        
        knowledge_used = []
        if knowledge.get("disease_info"):
            knowledge_used.append(knowledge["disease_info"].get("name", ""))
        if knowledge.get("possible_diseases"):
            knowledge_used.extend(knowledge["possible_diseases"])
        
        result_data = ConsultWithKnowledgeResultData(
            answer=answer,
            knowledge_used=knowledge_used,
            confidence=0.85 if knowledge_used else 0.5
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[ConsultWithKnowledgeChain] Chain执行完成: session_id={chain_context.session_id}, "
                   f"knowledge_used={len(knowledge_used)}, confidence={result_data.confidence}, elapsed={elapsed:.2f}s")
        
        return ChainResult(session_id=chain_context.session_id, data=result_data)
    
    def _build_knowledge_context(self, knowledge: Dict[str, Any]) -> str:
        """
        构建知识上下文字符串
        
        Args:
            knowledge: 知识字典
            
        Returns:
            知识上下文字符串
        """
        context_parts = []
        
        if knowledge.get("disease_info"):
            info = knowledge["disease_info"]
            context_parts.append(f"疾病名称：{info.get('name', '')}")
            if info.get('desc'):
                context_parts.append(f"描述：{info['desc']}")
            if info.get('cause'):
                context_parts.append(f"病因：{info['cause']}")
            if info.get('prevent'):
                context_parts.append(f"预防：{info['prevent']}")
        
        if knowledge.get("symptoms"):
            context_parts.append(f"症状：{', '.join(knowledge['symptoms'][:5])}")
        
        if knowledge.get("drugs"):
            drugs_info = knowledge["drugs"]
            if drugs_info.get("common_drugs"):
                context_parts.append(f"常用药物：{', '.join(drugs_info['common_drugs'][:5])}")
        
        if knowledge.get("possible_diseases"):
            context_parts.append(f"可能的疾病：{', '.join(knowledge['possible_diseases'][:5])}")
        
        return "\n".join(context_parts) if context_parts else "未找到相关知识"
