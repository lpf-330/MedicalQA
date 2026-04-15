"""
MedicalEntityVector 综合向量数据库部署脚本
整合实体名称、属性、关系三种向量的完整部署流程
"""

import os
import sys
import time
import subprocess
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from pymilvus import connections, Collection, utility

from config import ZILLIZ_CONFIG, NEO4J_CONFIG, LOCAL_MODEL_CONFIG, MILVUS_CONFIG
from logger import get_logger, log_deployment_step, log_deployment_success, log_deployment_failure


@dataclass
class DeploymentStep:
    step_id: int
    name: str
    description: str
    module_name: str = ""
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveDeploymentOrchestrator:
    def __init__(
        self,
        auto_continue: bool = False,
        skip_entity: bool = False,
        skip_attribute: bool = False,
        skip_relation: bool = False
    ):
        self.logger = get_logger()
        self.auto_continue = auto_continue
        self.skip_entity = skip_entity
        self.skip_attribute = skip_attribute
        self.skip_relation = skip_relation
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.steps: List[DeploymentStep] = []
        self.current_step_idx = 0
        self.failed_steps: List[int] = []
        self.report_dir = "logs"
        
        self.created_collections: List[str] = []
        
        self._initialize_steps()
        self._ensure_report_dir()
    
    def _initialize_steps(self):
        self.steps = [
            DeploymentStep(
                step_id=1,
                name="环境检查",
                description="检查Python环境和依赖包是否安装完整"
            ),
            DeploymentStep(
                step_id=2,
                name="配置验证",
                description="验证Neo4j、Zilliz Cloud等关键配置是否正确"
            ),
            DeploymentStep(
                step_id=3,
                name="实体名称向量部署",
                description="部署medical_entity集合（实体名称向量）",
                module_name="deploy_entity"
            ),
            DeploymentStep(
                step_id=4,
                name="实体属性向量部署",
                description="部署entity_attributes集合（实体属性向量）",
                module_name="deploy_attribute"
            ),
            DeploymentStep(
                step_id=5,
                name="关系向量部署",
                description="部署entity_relations集合（实体关系向量）",
                module_name="deploy_relation"
            ),
            DeploymentStep(
                step_id=6,
                name="部署验证",
                description="验证所有集合的部署结果和数据质量"
            )
        ]
    
    def _ensure_report_dir(self):
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir, exist_ok=True)
    
    def _print_banner(self):
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              MedicalEntityVector 综合向量数据库部署系统                        ║
║           Comprehensive Vector Database Deployment System                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        self.logger.info(banner)
    
    def _print_step_header(self, step: DeploymentStep):
        header = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
步骤 {step.step_id}/{len(self.steps)}: {step.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
描述: {step.description}
        """
        print(header)
        self.logger.info(f"开始执行步骤 {step.step_id}: {step.name}")
    
    def _print_progress_bar(
        self,
        current: int,
        total: int,
        description: str = "",
        bar_length: int = 50
    ):
        if total == 0:
            return
        
        progress = current / total
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        percentage = progress * 100
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        if current > 0 and elapsed_time > 0:
            speed = current / elapsed_time
            remaining_count = total - current
            eta_seconds = remaining_count / speed if speed > 0 else 0
            eta_str = f"预计剩余: {int(eta_seconds // 60)}分{int(eta_seconds % 60)}秒"
        else:
            eta_str = "预计剩余: 计算中..."
        
        progress_line = f"\r进度: [{bar}] {percentage:.1f}% ({current}/{total})"
        if description:
            progress_line += f" | {description}"
        progress_line += f" | {eta_str}"
        
        print(progress_line, end='', flush=True)
    
    def check_environment(self) -> Tuple[bool, str]:
        log_deployment_step("检查Python环境", "开始")
        
        python_version = sys.version_info
        python_version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        self.logger.info(f"Python版本: {python_version_str}")
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            error_msg = f"Python版本过低，需要3.8+，当前版本: {python_version_str}"
            log_deployment_failure("检查Python环境", error_msg)
            return False, error_msg
        
        log_deployment_step("检查依赖包", "开始")
        
        required_packages = [
            ("neo4j", "neo4j"),
            ("pymilvus", "pymilvus"),
            ("sentence_transformers", "sentence_transformers"),
            ("torch", "torch"),
            ("numpy", "numpy")
        ]
        
        missing_packages = []
        
        for package_name, import_name in required_packages:
            try:
                importlib.import_module(import_name)
                self.logger.info(f"  ✓ {package_name} 已安装")
            except ImportError:
                missing_packages.append(package_name)
                self.logger.error(f"  ✗ {package_name} 未安装")
        
        if missing_packages:
            error_msg = f"缺少依赖包: {', '.join(missing_packages)}"
            log_deployment_failure("检查依赖包", error_msg)
            return False, error_msg
        
        log_deployment_success("检查Python环境")
        log_deployment_success("检查依赖包")
        
        return True, "环境检查通过"
    
    def validate_config(self) -> Tuple[bool, str]:
        log_deployment_step("验证配置文件", "开始")
        
        self.logger.info("检查Neo4j配置...")
        if not NEO4J_CONFIG.get("uri"):
            error_msg = "Neo4j URI未配置"
            log_deployment_failure("验证配置文件", error_msg)
            return False, error_msg
        self.logger.info(f"  ✓ Neo4j URI: {NEO4J_CONFIG['uri']}")
        
        self.logger.info("检查Zilliz Cloud配置...")
        if ZILLIZ_CONFIG.get("uri") == "YOUR_ZILLIZ_CLOUD_URI_PLACEHOLDER":
            error_msg = (
                "Zilliz Cloud URI未配置！\n"
                "请按以下步骤获取实际URI：\n"
                "1. 登录Zilliz Cloud控制台 (https://cloud.zilliz.com)\n"
                "2. 选择您的集群\n"
                "3. 在集群详情页面找到 'Public Endpoint' 或 'URI'\n"
                "4. 将config.py中的ZILLIZ_CONFIG['uri']替换为实际地址\n"
                "示例格式: https://inxx-xxxx.api.gcp-us-west1.zillizcloud.com"
            )
            log_deployment_failure("验证配置文件", error_msg)
            return False, error_msg
        
        self.logger.info(f"  ✓ Zilliz Cloud URI: {ZILLIZ_CONFIG['uri']}")
        self.logger.info(f"  ✓ Zilliz Cloud User: {ZILLIZ_CONFIG['user']}")
        
        self.logger.info("检查本地模型配置...")
        self.logger.info(f"  ✓ 模型名称: {LOCAL_MODEL_CONFIG['model_name']}")
        self.logger.info(f"  ✓ 向量维度: {LOCAL_MODEL_CONFIG['dimension']}")
        self.logger.info(f"  ✓ 批次大小: {LOCAL_MODEL_CONFIG['batch_size']}")
        
        log_deployment_success("验证配置文件")
        return True, "配置验证通过"
    
    def deploy_entity_vectors(self) -> Tuple[bool, str]:
        log_deployment_step("部署实体名称向量", "开始")
        
        try:
            self.logger.info("检查medical_entity集合是否存在...")
            
            connections.connect(
                alias="default",
                token=ZILLIZ_CONFIG["token"],
                uri=ZILLIZ_CONFIG["uri"]
            )
            
            if utility.has_collection("medical_entity"):
                self.logger.info("medical_entity集合已存在，跳过实体名称向量部署")
                log_deployment_success("部署实体名称向量（集合已存在）")
                connections.disconnect("default")
                return True, "集合已存在，跳过部署"
            
            connections.disconnect("default")
            
            self.logger.info("medical_entity集合不存在，开始部署...")
            
            steps = [
                ("create_collection.py", "创建medical_entity集合"),
                ("extract_entities.py", "提取实体数据"),
                ("generate_vectors_local.py", "生成实体向量"),
                ("import_vectors.py", "导入实体向量")
            ]
            
            for idx, (script_name, step_desc) in enumerate(steps, 1):
                self._print_progress_bar(idx - 1, len(steps), step_desc)
                
                if not os.path.exists(script_name):
                    error_msg = f"脚本不存在: {script_name}"
                    log_deployment_failure(f"部署实体名称向量 - {step_desc}", error_msg)
                    return False, error_msg
                
                self.logger.info(f"\n执行: {script_name} - {step_desc}")
                
                result = subprocess.run(
                    [sys.executable, script_name],
                    capture_output=True,
                    text=True,
                    timeout=7200
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr if result.stderr else f"{script_name}执行失败"
                    log_deployment_failure(f"部署实体名称向量 - {step_desc}", error_msg)
                    return False, error_msg
                
                self.logger.info(f"{script_name} 执行成功")
            
            self._print_progress_bar(len(steps), len(steps), "完成")
            print()
            
            self.created_collections.append("medical_entity")
            
            log_deployment_success("部署实体名称向量")
            return True, "实体名称向量部署成功"
            
        except Exception as e:
            error_msg = f"部署实体名称向量失败: {str(e)}"
            log_deployment_failure("部署实体名称向量", error_msg)
            return False, error_msg
    
    def deploy_attribute_vectors(self) -> Tuple[bool, str]:
        log_deployment_step("部署实体属性向量", "开始")
        
        try:
            self.logger.info("开始部署entity_attributes集合...")
            
            steps = [
                ("create_entity_attributes_collection.py", "创建entity_attributes集合"),
                ("extract_entity_attributes.py", "提取实体属性数据"),
                ("generate_attribute_vectors.py", "生成并导入属性向量")
            ]
            
            for idx, (script_name, step_desc) in enumerate(steps, 1):
                self._print_progress_bar(idx - 1, len(steps), step_desc)
                
                if not os.path.exists(script_name):
                    error_msg = f"脚本不存在: {script_name}"
                    log_deployment_failure(f"部署实体属性向量 - {step_desc}", error_msg)
                    return False, error_msg
                
                self.logger.info(f"\n执行: {script_name} - {step_desc}")
                
                result = subprocess.run(
                    [sys.executable, script_name],
                    capture_output=True,
                    text=True,
                    timeout=7200
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr if result.stderr else f"{script_name}执行失败"
                    log_deployment_failure(f"部署实体属性向量 - {step_desc}", error_msg)
                    return False, error_msg
                
                self.logger.info(f"{script_name} 执行成功")
            
            self._print_progress_bar(len(steps), len(steps), "完成")
            print()
            
            self.created_collections.append("entity_attributes")
            
            log_deployment_success("部署实体属性向量")
            return True, "实体属性向量部署成功"
            
        except Exception as e:
            error_msg = f"部署实体属性向量失败: {str(e)}"
            log_deployment_failure("部署实体属性向量", error_msg)
            return False, error_msg
    
    def deploy_relation_vectors(self) -> Tuple[bool, str]:
        log_deployment_step("部署实体关系向量", "开始")
        
        try:
            self.logger.info("开始部署entity_relations集合...")
            
            steps = [
                ("create_entity_relations_collection.py", "创建entity_relations集合"),
                ("extract_relations.py", "提取实体关系数据"),
                ("generate_relation_vectors.py", "生成并导入关系向量")
            ]
            
            for idx, (script_name, step_desc) in enumerate(steps, 1):
                self._print_progress_bar(idx - 1, len(steps), step_desc)
                
                if not os.path.exists(script_name):
                    error_msg = f"脚本不存在: {script_name}"
                    log_deployment_failure(f"部署实体关系向量 - {step_desc}", error_msg)
                    return False, error_msg
                
                self.logger.info(f"\n执行: {script_name} - {step_desc}")
                
                result = subprocess.run(
                    [sys.executable, script_name],
                    capture_output=True,
                    text=True,
                    timeout=7200
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr if result.stderr else f"{script_name}执行失败"
                    log_deployment_failure(f"部署实体关系向量 - {step_desc}", error_msg)
                    return False, error_msg
                
                self.logger.info(f"{script_name} 执行成功")
            
            self._print_progress_bar(len(steps), len(steps), "完成")
            print()
            
            self.created_collections.append("entity_relations")
            
            log_deployment_success("部署实体关系向量")
            return True, "实体关系向量部署成功"
            
        except Exception as e:
            error_msg = f"部署实体关系向量失败: {str(e)}"
            log_deployment_failure("部署实体关系向量", error_msg)
            return False, error_msg
    
    def verify_deployment(self) -> Tuple[bool, str]:
        log_deployment_step("验证部署结果", "开始")
        
        try:
            connections.connect(
                alias="default",
                token=ZILLIZ_CONFIG["token"],
                uri=ZILLIZ_CONFIG["uri"]
            )
            
            expected_collections = []
            if not self.skip_entity:
                expected_collections.append("medical_entity")
            if not self.skip_attribute:
                expected_collections.append("entity_attributes")
            if not self.skip_relation:
                expected_collections.append("entity_relations")
            
            verification_results = {}
            all_passed = True
            
            for collection_name in expected_collections:
                self.logger.info(f"\n验证集合: {collection_name}")
                
                if not utility.has_collection(collection_name):
                    self.logger.error(f"  ✗ 集合不存在")
                    verification_results[collection_name] = {
                        "status": "failed",
                        "reason": "集合不存在"
                    }
                    all_passed = False
                    continue
                
                collection = Collection(collection_name)
                collection.load()
                
                entity_count = collection.num_entities
                self.logger.info(f"  ✓ 集合存在")
                self.logger.info(f"  ✓ 实体数量: {entity_count}")
                
                indexes = collection.indexes
                if indexes:
                    for index in indexes:
                        self.logger.info(f"  ✓ 索引字段: {index.field_name}")
                        self.logger.info(f"    索引类型: {index.params.get('index_type')}")
                        self.logger.info(f"    度量类型: {index.params.get('metric_type')}")
                else:
                    self.logger.warning(f"  ⚠ 未找到索引")
                
                verification_results[collection_name] = {
                    "status": "success",
                    "entity_count": entity_count,
                    "has_index": len(indexes) > 0
                }
            
            connections.disconnect("default")
            
            if all_passed:
                log_deployment_success("验证部署结果")
                return True, "所有集合验证通过"
            else:
                failed_collections = [
                    name for name, result in verification_results.items()
                    if result["status"] == "failed"
                ]
                error_msg = f"部分集合验证失败: {', '.join(failed_collections)}"
                log_deployment_failure("验证部署结果", error_msg)
                return False, error_msg
            
        except Exception as e:
            error_msg = f"验证部署结果失败: {str(e)}"
            log_deployment_failure("验证部署结果", error_msg)
            return False, error_msg
    
    def rollback_deployment(self) -> Tuple[bool, str]:
        log_deployment_step("回滚部署", "开始")
        
        try:
            self.logger.info("开始回滚部署，删除已创建的集合...")
            
            connections.connect(
                alias="default",
                token=ZILLIZ_CONFIG["token"],
                uri=ZILLIZ_CONFIG["uri"]
            )
            
            collections_to_drop = []
            
            if not self.skip_entity and utility.has_collection("medical_entity"):
                collections_to_drop.append("medical_entity")
            if not self.skip_attribute and utility.has_collection("entity_attributes"):
                collections_to_drop.append("entity_attributes")
            if not self.skip_relation and utility.has_collection("entity_relations"):
                collections_to_drop.append("entity_relations")
            
            if not collections_to_drop:
                self.logger.info("没有需要删除的集合")
                connections.disconnect("default")
                return True, "没有需要回滚的集合"
            
            for collection_name in collections_to_drop:
                self.logger.info(f"删除集合: {collection_name}")
                utility.drop_collection(collection_name)
                self.logger.info(f"  ✓ 集合 {collection_name} 已删除")
            
            connections.disconnect("default")
            
            log_deployment_success("回滚部署")
            return True, f"成功删除 {len(collections_to_drop)} 个集合"
            
        except Exception as e:
            error_msg = f"回滚部署失败: {str(e)}"
            log_deployment_failure("回滚部署", error_msg)
            return False, error_msg
    
    def _ask_continue(self, step: DeploymentStep) -> bool:
        if self.auto_continue:
            self.logger.warning(f"自动继续模式：跳过失败步骤 {step.name}")
            return True
        
        print(f"\n{'='*60}")
        print(f"步骤 '{step.name}' 执行失败！")
        print(f"错误信息: {step.error_message}")
        print(f"{'='*60}")
        print("\n请选择操作:")
        print("  [C] 继续执行下一步")
        print("  [R] 重试当前步骤")
        print("  [A] 中止部署")
        print("  [B] 回滚部署并退出")
        
        while True:
            choice = input("\n请输入选择 (C/R/A/B): ").strip().upper()
            
            if choice == 'C':
                self.logger.info(f"用户选择继续执行，跳过步骤 {step.name}")
                return True
            elif choice == 'R':
                self.logger.info(f"用户选择重试步骤 {step.name}")
                return self._execute_step(step)
            elif choice == 'A':
                self.logger.info("用户选择中止部署")
                return False
            elif choice == 'B':
                self.logger.info("用户选择回滚部署并退出")
                self.rollback_deployment()
                return False
            else:
                print("无效选择，请重新输入")
    
    def _execute_step(self, step: DeploymentStep) -> bool:
        step.start_time = time.time()
        step.status = "running"
        
        self._print_step_header(step)
        
        success = False
        message = ""
        
        if step.step_id == 1:
            success, message = self.check_environment()
        elif step.step_id == 2:
            success, message = self.validate_config()
        elif step.step_id == 3:
            if self.skip_entity:
                self.logger.info("跳过实体名称向量部署（--skip-entity）")
                step.status = "skipped"
                step.details["message"] = "用户指定跳过"
                return True
            success, message = self.deploy_entity_vectors()
        elif step.step_id == 4:
            if self.skip_attribute:
                self.logger.info("跳过实体属性向量部署（--skip-attribute）")
                step.status = "skipped"
                step.details["message"] = "用户指定跳过"
                return True
            success, message = self.deploy_attribute_vectors()
        elif step.step_id == 5:
            if self.skip_relation:
                self.logger.info("跳过关系向量部署（--skip-relation）")
                step.status = "skipped"
                step.details["message"] = "用户指定跳过"
                return True
            success, message = self.deploy_relation_vectors()
        elif step.step_id == 6:
            success, message = self.verify_deployment()
        
        step.end_time = time.time()
        step.duration = step.end_time - step.start_time
        
        if success:
            step.status = "success"
            step.details["message"] = message
            self.logger.info(f"步骤 {step.step_id} 完成，耗时: {step.duration:.2f} 秒")
            return True
        else:
            step.status = "failed"
            step.error_message = message
            self.failed_steps.append(step.step_id)
            self.logger.error(f"步骤 {step.step_id} 失败: {message}")
            return False
    
    def generate_deployment_report(self) -> str:
        self.logger.info("生成部署报告...")
        
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MedicalEntityVector 综合向量数据库部署报告")
        report_lines.append("=" * 80)
        report_lines.append(f"部署时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
        report_lines.append("")
        
        report_lines.append("-" * 80)
        report_lines.append("部署配置")
        report_lines.append("-" * 80)
        report_lines.append(f"跳过实体名称向量: {'是' if self.skip_entity else '否'}")
        report_lines.append(f"跳过实体属性向量: {'是' if self.skip_attribute else '否'}")
        report_lines.append(f"跳过关系向量: {'是' if self.skip_relation else '否'}")
        report_lines.append("")
        
        report_lines.append("-" * 80)
        report_lines.append("部署步骤详情")
        report_lines.append("-" * 80)
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for step in self.steps:
            status_icon = "✓" if step.status == "success" else ("✗" if step.status == "failed" else "○")
            status_text = "成功" if step.status == "success" else ("失败" if step.status == "failed" else "跳过")
            
            report_lines.append(f"\n步骤 {step.step_id}: {step.name}")
            report_lines.append(f"  状态: {status_icon} {status_text}")
            report_lines.append(f"  描述: {step.description}")
            
            if step.duration > 0:
                report_lines.append(f"  耗时: {step.duration:.2f} 秒")
            
            if step.status == "success":
                success_count += 1
                if step.details.get("message"):
                    report_lines.append(f"  结果: {step.details['message']}")
            elif step.status == "failed":
                failed_count += 1
                report_lines.append(f"  错误: {step.error_message}")
            else:
                skipped_count += 1
        
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("部署统计")
        report_lines.append("-" * 80)
        report_lines.append(f"总步骤数: {len(self.steps)}")
        report_lines.append(f"成功步骤: {success_count}")
        report_lines.append(f"失败步骤: {failed_count}")
        report_lines.append(f"跳过步骤: {skipped_count}")
        report_lines.append(f"成功率: {success_count / len(self.steps) * 100:.1f}%")
        
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("已创建集合")
        report_lines.append("-" * 80)
        if self.created_collections:
            for collection_name in self.created_collections:
                report_lines.append(f"  - {collection_name}")
        else:
            report_lines.append("  无")
        
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("部署结果")
        report_lines.append("-" * 80)
        
        if failed_count == 0 and success_count == len(self.steps) - skipped_count:
            report_lines.append("✓ 部署成功！所有步骤均已成功执行。")
            report_lines.append("")
            report_lines.append("向量数据库已成功部署，可以进行以下操作:")
            report_lines.append("  1. 使用 verify_deployment.py 验证部署状态")
            report_lines.append("  2. 开始使用向量检索功能")
            report_lines.append("  3. 测试混合检索功能")
        elif failed_count > 0:
            report_lines.append("✗ 部署失败！以下步骤执行失败:")
            for step_id in self.failed_steps:
                step = self.steps[step_id - 1]
                report_lines.append(f"  - 步骤 {step_id}: {step.name}")
                report_lines.append(f"    原因: {step.error_message}")
            report_lines.append("")
            report_lines.append("建议操作:")
            report_lines.append("  1. 检查错误日志了解详细错误信息")
            report_lines.append("  2. 修复问题后重新运行部署脚本")
            report_lines.append("  3. 或使用 --rollback 参数回滚部署")
        else:
            report_lines.append("⚠ 部署部分完成，部分步骤被跳过。")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        return report
    
    def save_report(self, report: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.report_dir, f"comprehensive_deployment_report_{timestamp}.txt")
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"部署报告已保存: {report_file}")
            return report_file
        except Exception as e:
            self.logger.error(f"保存部署报告失败: {str(e)}")
            return ""
    
    def run(self) -> bool:
        self._print_banner()
        
        self.logger.info(f"开始部署时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"总步骤数: {len(self.steps)}")
        self.logger.info(f"自动继续模式: {'是' if self.auto_continue else '否'}")
        self.logger.info(f"跳过实体名称向量: {'是' if self.skip_entity else '否'}")
        self.logger.info(f"跳过实体属性向量: {'是' if self.skip_attribute else '否'}")
        self.logger.info(f"跳过关系向量: {'是' if self.skip_relation else '否'}")
        
        self.start_time = time.time()
        
        for idx, step in enumerate(self.steps):
            self.current_step_idx = idx
            
            success = self._execute_step(step)
            
            if not success:
                should_continue = self._ask_continue(step)
                
                if not should_continue:
                    for remaining_step in self.steps[idx + 1:]:
                        remaining_step.status = "skipped"
                    break
        
        self.end_time = time.time()
        
        report = self.generate_deployment_report()
        report_file = self.save_report(report)
        
        print("\n" + report)
        
        if report_file:
            print(f"\n部署报告已保存至: {report_file}")
        
        return len(self.failed_steps) == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MedicalEntityVector 综合向量数据库部署脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python deploy_comprehensive.py                    # 完整部署（实体、属性、关系）
  python deploy_comprehensive.py --skip-entity      # 跳过实体名称向量部署
  python deploy_comprehensive.py --skip-attribute   # 跳过实体属性向量部署
  python deploy_comprehensive.py --skip-relation    # 跳过关系向量部署
  python deploy_comprehensive.py --auto             # 自动部署，失败时自动跳过
  python deploy_comprehensive.py --rollback         # 回滚部署（删除所有集合）
        """
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动继续模式，失败时自动跳过继续执行"
    )
    
    parser.add_argument(
        "--skip-entity",
        action="store_true",
        help="跳过实体名称向量部署"
    )
    
    parser.add_argument(
        "--skip-attribute",
        action="store_true",
        help="跳过实体属性向量部署"
    )
    
    parser.add_argument(
        "--skip-relation",
        action="store_true",
        help="跳过关系向量部署"
    )
    
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="回滚部署（删除所有已创建的集合）"
    )
    
    args = parser.parse_args()
    
    if args.rollback:
        print("=" * 60)
        print("警告：即将删除所有向量集合！")
        print("=" * 60)
        print("\n将删除以下集合（如果存在）:")
        if not args.skip_entity:
            print("  - medical_entity")
        if not args.skip_attribute:
            print("  - entity_attributes")
        if not args.skip_relation:
            print("  - entity_relations")
        
        confirm = input("\n确认删除？(yes/no): ").strip().lower()
        
        if confirm == "yes":
            orchestrator = ComprehensiveDeploymentOrchestrator(
                auto_continue=args.auto,
                skip_entity=args.skip_entity,
                skip_attribute=args.skip_attribute,
                skip_relation=args.skip_relation
            )
            success, message = orchestrator.rollback_deployment()
            
            if success:
                print(f"\n✓ 回滚成功！{message}")
                return 0
            else:
                print(f"\n✗ 回滚失败！{message}")
                return 1
        else:
            print("\n已取消回滚操作")
            return 0
    
    orchestrator = ComprehensiveDeploymentOrchestrator(
        auto_continue=args.auto,
        skip_entity=args.skip_entity,
        skip_attribute=args.skip_attribute,
        skip_relation=args.skip_relation
    )
    
    success = orchestrator.run()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ 部署成功完成！")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ 部署失败，请查看报告了解详情")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
