# 输入数据格式规范（LLM 打谱，音位表增强版）

## A. YOLO 原始输出（模块A）
- 类型：`List[Dict]`
- 关键字段：`class`, `bbox=[x1,y1,x2,y2]`, `conf`
- 示例文件：`input_sample_01_yolo_boxes_12.json`

## B. Topology 聚合输出（模块B）
- 类型：`Dict[group_id -> payload]`
- 关键字段：`right_fingering`, `left_fingering`, `left_finger`, `hui`, `xian`, `group_bbox`, `components`
- 示例文件：`input_sample_02_topology_dict_first2.json`

## C. 喂给 LLM 的紧凑输入（模块C）
- 类型：`List[Dict]`（按 group 顺序）
- 基础字段：`group_id,right_fingering,left_fingering,left_finger,hui,xian,action,finger,position,string`
- 音位表增强字段：`tone_table_hit,tone_table_pitch,tone_table_octave,tone_table_ref`
- 规则：当 `tone_table_hit=true` 时，LLM 需优先采用 `tone_table_pitch/tone_table_octave`
- 示例文件：
  - `input_sample_03_llm_compact_list_first12.json`
  - `input_sample_04_user_list_style_first12.json`
  - `input_sample_05_compact_with_tone_table_hits.json`

## D. 音位表数据源
- 原始文件：`tone_table_source_音位表.xlsx`
- 转换后映射：`qin_tone_table_full.json`
- 摘要样例：`input_sample_06_tone_table_excerpt_first30.json`

## E. 当前系统约束
- `pitch` 取值：`"1"~"7"`
- `octave` 取值：`"3"|"4"|"5"`
- `duration` 取值：`"2"|"4"|"8"|"16"`
- `new_measure`：布尔值
