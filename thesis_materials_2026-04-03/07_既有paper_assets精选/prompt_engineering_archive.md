# Prompt Engineering 档案

## 1) 主提示词（Main Prompt）
- `system_prompt_main.txt`
- `user_prompt_main_example.txt`

## 2) 修复提示词（Repair Prompt）
- 当主输出字段越界（如 pitch=9）时触发
- `system_prompt_repair.txt`
- `user_prompt_repair_example.txt`

## 3) Few-Shot 现状
- 当前代码中未显式加入 few-shot 示例段（即没有固定示例对直接拼入主 Prompt）
- 目前主要通过“规则约束 + 输出校验 + 修复提示词”提升稳定性

## 4) 可写入论文的说明句
- 本研究采用“主生成 + 结构化校验 + 二次修复”的提示词策略，以降低复杂乐谱序列中字段越界问题。
