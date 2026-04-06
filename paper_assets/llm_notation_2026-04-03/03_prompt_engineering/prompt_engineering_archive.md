# Prompt Engineering 档案（音位表增强版）

## 1) 主提示词（Main Prompt）
- `system_prompt_main.txt`
- `user_prompt_main_example.txt`
- 关键新增：音位表命中规则
  - 当 `tone_table_hit=true` 时，优先采用 `tone_table_pitch/tone_table_octave`

## 2) 修复提示词（Repair Prompt）
- `system_prompt_repair.txt`
- `user_prompt_repair_example.txt`
- 修复阶段同样保留音位表优先策略，避免校正时覆盖有效映射

## 3) Few-Shot 现状
- 当前代码中未加入固定 few-shot 样例
- 当前稳定性策略：`规则约束 + 结构化输出 + 严格校验 + 修复提示词`

## 4) 论文可引用句
- 本研究在 LLM 打谱中引入“音位表增强提示”，将部分弦序-徽序直接映射为候选音高八度，并通过提示词优先级约束提升可解释性与稳定性。
