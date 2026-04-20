# 论文插图导出

## PNG 文件
- figure_3_9_semantic_reconstruction.png
- figure_3_11_prompt_repair_loop.png
- figure_5_4_case_study_rendering.png

## 关键片段
- snippets/figure_3_9_input_groups.json
- snippets/figure_3_9_output_notes.json
- snippets/figure_3_11_first_round_output.json
- snippets/figure_3_11_invalid_draft.json
- snippets/figure_3_11_repaired_output.json
- snippets/figure_3_11_validator_rules.txt
- snippets/figure_5_4_musicxml_snippet.xml
- snippets/figure_5_4_structured_notes.json

## 生成说明
- 图3-9 使用真实减字组样例与音位表命中结果重绘方法图，强调 X/K/P 三类约束如何共同完成语义重构。
- 图3-11 使用本地修复提示样例中的非法字段案例展示主生成、校验门与局部修复闭环。
- 图5-4 使用 testpicture-1 的原始输入、聚合结果、裁剪后的终态谱面渲染与 MusicXML 片段。
- 本次未重新运行 YOLO/上传链路，沿用线程中已保存的成功样例资产。
- 模型配置沿用当前后端：qwen3.5-plus，temperature=0.0。

## Caption / Alt Text 草案
- 图3-9 caption：中间表示、音位表知识和提示词约束共同限制 LLM 的减字谱语义重构过程，校验与修复模块进一步保证结构化输出合法。
- 图3-9 alt：图中从减字组序列 X、音位表 K 和提示词约束 P 三路输入进入 LLM 语义重构器，经本地校验和局部修复后输出标准化字段。
- 图3-11 caption：LLM 首轮生成结果先进入本地校验门，合法结果直接输出；非法字段被送入修复提示词进行局部修正，并再次校验。
- 图3-11 alt：闭环流程图展示减字组序列经主生成提示词得到 JSON，校验失败时生成错误反馈并进入修复提示词，修复后返回校验器。
- 图5-4 caption：代表性测试样例从原始减字谱输入、减字组检测聚合，到终态谱面渲染与 MusicXML 标准化片段的实测链路。
- 图5-4 alt：四面板图展示同一测试图片的原始谱例、红框聚合检测结果、裁剪后的现代谱面渲染和对应 MusicXML 代码片段。
