# 转换效果实证摘要（音位表增强版）

- 测试图片：`testpicture-1.jpg`
- 当前模型：`qwen3.5-plus`
- MusicXML 生成：`是`
- 失败策略：LLM 调用/解析异常时严格报错（不走规则兜底）

## 音位表增强证据
- 音位表原件：`../01_input_data_spec/tone_table_source_音位表.xlsx`
- 音位表映射：`../01_input_data_spec/qin_tone_table_full.json`
- LLM 输入命中样例：`../01_input_data_spec/input_sample_05_compact_with_tone_table_hits.json`
- 关键提示词约束：命中 `tone_table_hit=true` 时优先采用映射音高

## 对应素材
- LLM 原始输出片段：`llm_output_sample_first30_notes.json`
- MusicXML 全量：`musicxml_full_from_testpicture.xml`
- MusicXML 片段：`musicxml_snippet_head_120_lines.xml`
- 渲染截图：`final_rendered_result.png`、`scoremodel-after-fix.png`
- 检测/聚合可视化：`testpicture-1_red_boxes.jpg`、`testpicture-1_group_boxes_only_v3.jpg`
