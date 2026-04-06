# LLM 打谱论文素材包

> 最近更新：2026-04-04 20:54:21
> 根目录：`F:\AIcharacter\End\paper_assets\llm_notation_2026-04-03`

## 目录说明
- `01_input_data_spec/`：输入格式规范、YOLO/Topology/LLM 输入样例、音位表源文件与映射样例
- `02_model_and_config/`：模型选型、参数配置与“音位表增强提示”配置记录
- `03_prompt_engineering/`：主提示词、修复提示词、主/修复用户提示样例
- `04_musicxml_mapping/`：减字谱元素到 MusicXML 的映射规则
- `05_conversion_evidence/`：LLM 输出、MusicXML 片段、渲染结果与实证说明
- `06_pipeline_diagram/`：全流程文字化管线说明（含音位表增强步骤）
- `07_extra_materials/`：顺序核查图与测试图
- `08_source_snapshots/`：关键源码快照（已同步音位表版本）

## 本次新增（音位表接入）
- 新增音位表原件副本：`01_input_data_spec/tone_table_source_音位表.xlsx`
- 新增音位表 JSON：`01_input_data_spec/qin_tone_table_full.json`
- 新增 LLM 输入命中样例：`01_input_data_spec/input_sample_05_compact_with_tone_table_hits.json`
- 提示词已加入规则：`tone_table_hit=true` 时优先采用 `tone_table_pitch/tone_table_octave`
- 源码快照已更新：`08_source_snapshots/llm_module.py`、`08_source_snapshots/qin_tone_table.json`
