---
title: "Axolotl 训练指南"
sidebar_label: "Axolotl 训练"
---

# Axolotl 训练

此页面为 `mlops-training-axolotl` 的中文页面，简要说明训练流程与外部依赖：

- 准备数据集并进行格式化（JSONL / Parquet）。
- 使用 Axolotl 的训练脚本启动分布式训练，配置 checkpoint 与混合精度。 
- 常见调优项：学习率调度、微调阶段冻结层、batch 大小与梯度累积。

此页面用于消除构建中对 `mlops-training-axolotl` 的缺页告警，详细内容请参考英文原始页面或技能库。

