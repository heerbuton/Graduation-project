# LLM 打谱流程说明文档（音位表增强版）

## 1. 总体流程
1. 模块A：YOLO 识别减字符号与坐标
2. 模块B：按版面顺序聚合为 group 拓扑
3. 模块C：构造 LLM 输入（含 `tone_table_*` 提示字段）
4. 模块C：调用 `qwen3.5-plus` 输出结构化 `notes`
5. 模块C：严格校验；必要时触发修复提示词
6. 模块D：映射为 MusicXML 并渲染现代谱

## 2. 音位表增强步骤（新增）
- 数据源：`相关文档/音位表.xlsx`
- 转换文件：`backend/pipeline/qin_tone_table.json`
- 运行时规则：
  - 对每个 group 解析 `xian/hui`
  - 若命中映射，注入：
    - `tone_table_hit=true`
    - `tone_table_pitch`
    - `tone_table_octave`
    - `tone_table_ref`
  - 主提示词要求：命中时优先采用映射音高八度

## 3. 输出结构约束
- `pitch`: `"1"~"7"`
- `octave`: `"3"|"4"|"5"`
- `duration`: `"2"|"4"|"8"|"16"`
- `new_measure`: `bool`
- `action/string/position/finger`：保留输入原值

## 4. 论文可引用结论
- 音位表增强把“部分可确定音”前置为显式知识，使 LLM 从“纯推断”转为“知识约束下推断”。
- 该设计提高了可解释性：每个命中项可通过 `tone_table_ref` 回溯到弦序-徽序依据。
