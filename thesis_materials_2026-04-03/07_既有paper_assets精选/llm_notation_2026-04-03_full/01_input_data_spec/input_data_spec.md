# 输入数据格式规范（LLM 打谱）

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
- 字段：`group_id,right_fingering,left_fingering,left_finger,hui,xian,action,finger,position,string`
- 示例文件：
  - `input_sample_03_llm_compact_list_first12.json`
  - `input_sample_04_user_list_style_first12.json`

## D. 当前样本统计（testpicture-1.jpg）
- YOLO 检测框数：`377`
- Topology group 数：`135`
- LLM 输出 note 数：`134`
