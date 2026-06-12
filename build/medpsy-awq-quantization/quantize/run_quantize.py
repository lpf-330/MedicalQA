# -*- coding: utf-8 -*-
"""
AWQ量化主脚本

使用AutoAWQ对MedPsy-4B进行4-bit量化。
支持从YAML配置文件加载参数，使用自定义校准数据集。
"""

import json
import logging
import os
import time
from datetime import datetime

import yaml
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_calibration_texts(data_path: str, max_samples: int, max_seq_len: int) -> list:
    logger.info(f"加载校准数据: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data.get("texts", [])
    if not texts:
        samples = data.get("samples", [])
        texts = [s.get("text", "") for s in samples if s.get("text")]

    logger.info(f"校准文本总量: {len(texts)}")

    if len(texts) > max_samples:
        texts = texts[:max_samples]
        logger.info(f"截取到 max_calib_samples={max_samples}")

    texts = [t[:max_seq_len] for t in texts]
    return texts


def run_quantization(config_path: str):
    start_time = time.time()
    config = load_config(config_path)

    model_path = config["model_path"]
    output_path = config["output_path"]
    calib_data_path = config["calibration_data_path"]
    quant_config = config["quant_config"]
    calib_config = config["calib_config"]
    model_config = config["model_config"]

    logger.info("=" * 60)
    logger.info("MedPsy-4B AWQ 量化")
    logger.info("=" * 60)
    logger.info(f"原模型路径: {model_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"量化参数: bits={quant_config['bits']}, group_size={quant_config['group_size']}, "
                f"zero_point={quant_config['zero_point']}, version={quant_config['version']}")
    logger.info(f"校准参数: max_calib_samples={calib_config['max_calib_samples']}, "
                f"max_calib_seq_len={calib_config['max_calib_seq_len']}")

    calib_texts = load_calibration_texts(
        calib_data_path,
        calib_config["max_calib_samples"],
        calib_config["max_calib_seq_len"]
    )

    if not calib_texts:
        logger.error("校准数据为空，无法量化！")
        return False

    logger.info("加载模型...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=model_config.get("trust_remote_code", True),
        safetensors=True,
        device_map=model_config.get("device_map", "auto"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    logger.info("开始量化...")
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": quant_config["zero_point"],
            "q_group_size": quant_config["group_size"],
            "w_bit": quant_config["bits"],
            "version": quant_config["version"],
        },
        calib_data=calib_texts,
    )

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"保存量化模型到: {output_path}")

    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    elapsed = time.time() - start_time
    logger.info(f"量化完成，耗时: {elapsed/60:.1f} 分钟")

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            total_size += os.path.getsize(fp)

    quant_info = {
        "timestamp": datetime.now().isoformat(),
        "source_model": model_path,
        "output_path": output_path,
        "quant_config": quant_config,
        "calib_config": calib_config,
        "calib_samples_count": len(calib_texts),
        "elapsed_seconds": round(elapsed, 1),
        "total_size_gb": round(total_size / (1024**3), 2),
        "config_file": config_path,
    }

    info_path = os.path.join(output_path, "quantization_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(quant_info, f, ensure_ascii=False, indent=2)

    logger.info(f"量化模型总大小: {quant_info['total_size_gb']} GB")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MedPsy-4B AWQ量化")
    parser.add_argument("--config", type=str, required=True, help="量化配置YAML文件路径")
    args = parser.parse_args()

    success = run_quantization(args.config)
    if not success:
        logger.error("量化失败！")
        exit(1)
