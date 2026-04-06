# LLM 打谱流程说明文档（替代流程图）

> 文档用途：替代原先的流程图文件，提供可直接写入论文正文/附录的文字化技术说明。  
> 适用版本：`qwen3.5-plus`，后端 `backend/pipeline/llm_module.py` 当前实现。

---

## 1. 任务定义与总体目标

本模块（LLM 打谱）目标是：

1. 接收前序模块输出的减字谱结构化序列（按页面阅读顺序完成排序）。
2. 在保持完整上下文的前提下，为每个减字组推断：`pitch`（音高）、`octave`（八度）、`duration`（时值）与 `new_measure`（小节切换标记）。
3. 输出结构化 JSON，供 MusicXML 编码模块直接消费。

核心原则：

- **上下文优先**：不能逐字孤立打谱，必须全局观察整段序列的句法、重复型、节奏走势。
- **结构约束**：输出严格受枚举值与字段约束，保证后续编码稳定。
- **工程可恢复**：若首轮输出越界，触发“修复提示词”二次校正，避免整链路失败。

---

## 2. 数据流详细说明（文字版 Pipeline）

### 2.1 模块 A（YOLO 识别）

输入：古琴谱图像（例如 `testpicture-1.jpg`）  
输出：检测框序列（`class`, `bbox`, `conf`）。

示例（简化）：

```json
[
  {"class": "历", "bbox": [x1, y1, x2, y2], "conf": 0.91},
  {"class": "六", "bbox": [x1, y1, x2, y2], "conf": 0.88}
]
```

### 2.2 模块 B（Topology 聚合）

输入：YOLO 检测框  
输出：分组字典（`group_1 ... group_n`），每组包含右/左手指法、弦位、徽位等信息。

示例（简化）：

```json
{
  "group_1": {
    "right_fingering": "历",
    "left_fingering": "",
    "left_finger": "",
    "hui": "",
    "xian": "六",
    "action": "历",
    "finger": "",
    "position": "",
    "string": "六"
  }
}
```

### 2.3 模块 C（LLM 打谱）

输入：Topology 的完整序列（可为 dict 或 list，最终标准化为 list）。  
处理：

1. 规范化 group 结构。
2. 构造主提示词（System + User，包含全局规则）。
3. 调用 LLM 生成 `notes`。
4. 本地严格校验（字段、顺序、枚举值）。
5. 若失败，触发修复提示词再次请求。

输出：

```json
[
  {
    "pitch": "6",
    "octave": "3",
    "duration": "8",
    "action": "历",
    "string": "六",
    "position": "",
    "finger": "",
    "new_measure": false
  }
]
```

### 2.4 模块 D（MusicXML 编码）

将上述结构映射到 MusicXML：

- `pitch` -> `<pitch><step>...</step></pitch>`（经数字到 CDEFGAB 映射）
- `octave` -> `<octave>`
- `duration` -> `<duration>` + `<type>`
- `action/string/position/finger` -> 四行 `<lyric number="1~4">`
- `new_measure=true` -> 新 `<measure>`

---

## 3. 字段约束（论文可引用）

LLM 输出 `notes` 中每个对象必须满足：

- `group_id`: `string`
- `pitch`: `"1"|"2"|"3"|"4"|"5"|"6"|"7"`
- `octave`: `"3"|"4"|"5"`
- `duration`: `"2"|"4"|"8"|"16"`
- `action/string/position/finger`: `string`（来自输入同组字段，原样保留）
- `new_measure`: `boolean`

顺序约束：

- 输出长度与输入组数必须一致。
- `group_id` 必须与输入一一对应，且顺序一致。

---

## 4. 模型配置（论文可引用）

- 模型：`qwen3.5-plus`
- API：`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- `temperature`: `0.0`
- `enable_thinking`: `false`
- `response_format`: `json_object`
- 超时：`180s`

配置意图：

- 低温度提升确定性与复现性。
- 关闭思考模式降低延迟与不稳定输出的概率。
- 采用 JSON 对象输出并在本地做严格校验，兼顾速度与安全。

---

## 5. 提示词工程（丰富完整版）

以下为可直接落地的“论文版提示词模板”，建议作为附录给出。

### 5.1 主 System Prompt（完整版）

```text
你是古琴减字谱正调打谱专家，兼具中国古代乐理、减字谱语法、现代记谱法（简谱与MusicXML）知识。

【任务目标】
给定一段按阅读顺序排列的减字组序列（group_1...group_n），你必须基于整段上下文进行整体打谱。
禁止逐group独立机械判断，必须先做全局节奏与句法判断，再输出每个group的音高与时值。

【输出格式】
你只能输出一个JSON对象，且顶层仅有一个字段：notes。
notes是数组，长度必须等于输入group数量，且顺序与输入完全一致。
notes每个元素必须包含且仅包含以下字段：
- group_id
- pitch
- octave
- duration
- action
- string
- position
- finger
- new_measure

【硬性约束】
1) pitch 只能是字符串："1"~"7"，严禁"0"、"8"、"9"或其他值。
2) octave 只能是"3"、"4"、"5"。
3) duration 只能是"2"、"4"、"8"、"16"。
4) new_measure 必须是布尔值 true/false。
5) action/string/position/finger 必须原样继承输入中同group字段，不得翻译、拼音化、数字化或改写。
6) 每个输入group恰好对应一个输出note，不允许丢失、重复或重排group_id。

【古琴打谱规则】
A. 全局优先：先判断整段的句法与重复动机，再进行逐组落点。
B. 一致性：相同或高度相似的“指法-弦位-徽位”组合在相近语境下应保持音高/时值一致。
C. 近邻平滑：若证据不足，优先参考前后邻近组，避免无依据大幅跳变。
D. 节奏连贯：时值分配需满足乐句连贯，不可随机插入异常长短时值。
E. 小节标注：new_measure只能在结构性边界设置，不能频繁或随意出现。

【缺省回退规则】
当证据不足时，使用：pitch="1"，octave="4"，duration="4"。

【禁止事项】
- 禁止输出任何解释、分析、注释、Markdown代码块。
- 禁止输出除JSON对象以外的任何字符。
```

### 5.2 主 User Prompt 模板（完整版）

```text
以下是第二模块输出的完整减字序列。请基于完整上下文进行打谱，严格按输入顺序输出JSON：

{groups_json}
```

其中 `{groups_json}` 为紧凑 JSON（建议字段：`group_id,right_fingering,left_fingering,left_finger,hui,xian,action,finger,position,string`）。

### 5.3 Few-Shot 示例（可选）

> 如果后续需要进一步提升稳定性，可在主提示词后追加1~2个短示例（避免过长导致超时）。

#### 示例 1
输入：

```json
[
  {
    "group_id": "group_1",
    "right_fingering": "历",
    "left_fingering": "",
    "left_finger": "",
    "hui": "",
    "xian": "六",
    "action": "历",
    "finger": "",
    "position": "",
    "string": "六"
  }
]
```

输出：

```json
{
  "notes": [
    {
      "group_id": "group_1",
      "pitch": "6",
      "octave": "3",
      "duration": "8",
      "action": "历",
      "string": "六",
      "position": "",
      "finger": "",
      "new_measure": false
    }
  ]
}
```

#### 示例 2
输入：

```json
[
  {
    "group_id": "group_1",
    "right_fingering": "勾",
    "left_fingering": "",
    "left_finger": "食",
    "hui": "七",
    "xian": "一",
    "action": "勾",
    "finger": "食",
    "position": "七",
    "string": "一"
  },
  {
    "group_id": "group_2",
    "right_fingering": "抹",
    "left_fingering": "",
    "left_finger": "大",
    "hui": "七",
    "xian": "四",
    "action": "抹",
    "finger": "大",
    "position": "七",
    "string": "四"
  }
]
```

输出：

```json
{
  "notes": [
    {
      "group_id": "group_1",
      "pitch": "1",
      "octave": "4",
      "duration": "4",
      "action": "勾",
      "string": "一",
      "position": "七",
      "finger": "食",
      "new_measure": false
    },
    {
      "group_id": "group_2",
      "pitch": "4",
      "octave": "4",
      "duration": "4",
      "action": "抹",
      "string": "四",
      "position": "七",
      "finger": "大",
      "new_measure": false
    }
  ]
}
```

### 5.4 修复 System Prompt（用于二次校正）

```text
你是古琴打谱结果修复器。
你将收到输入groups和一份draft_notes，请仅修复draft中的不合法字段并输出合法JSON对象：{"notes":[...]}。

要求：
- notes顺序必须与groups一致。
- group_id必须一一对应，不得新增、删除、重排。
- pitch只能"1"~"7"；octave只能"3"/"4"/"5"；duration只能"2"/"4"/"8"/"16"；new_measure必须布尔值。
- action/string/position/finger必须与输入对应group保持一致，不得改写。
- 若无法确定，用默认值：pitch="1", octave="4", duration="4"。
- 只输出JSON，不要任何解释。
```

---

## 6. 论文写作可直接引用段落（建议）

### 6.1 方法描述段（可放“方法/实现”章节）

本研究在 LLM 打谱阶段采用“完整上下文推断 + 结构化约束 + 二次修复”策略。首先将 YOLO 检测与拓扑聚合结果规范化为按阅读顺序排列的 group 序列，并以紧凑 JSON 形式输入大模型。主提示词明确要求模型在全局上下文下完成音高与时值推断，且输出必须与输入组一一对齐。随后，系统对模型输出执行本地严格校验（字段完整性、枚举合法性、顺序一致性）；若出现越界值或错位结果，则触发修复提示词进行一次受控重写。该机制在保证打谱结果一致性的同时，显著降低了自由生成导致的结构错误风险。

### 6.2 工程稳定性段（可放“实验与讨论”章节）

为平衡推理质量与响应稳定性，系统使用 `qwen3.5-plus` 并固定 `temperature=0.0`、`enable_thinking=false`。输出格式采用 JSON 对象，并通过后处理校验将语义错误显式化，避免错误传播至 MusicXML 编码阶段。实验中该策略可稳定生成可渲染的结构化输出，并支持异常场景下的自动修复回路。

---

## 7. 与现有素材包的衔接

建议与以下文件联合引用：

- 输入样例：`01_input_data_spec/input_sample_03_llm_compact_list_first12.json`
- Prompt 档案：`03_prompt_engineering/`
- 映射规则：`04_musicxml_mapping/musicxml_mapping_rules_table.md`
- 实证输出：`05_conversion_evidence/llm_output_sample_first30_notes.json`

---

## 8. 版本说明

- 本文档用于替代原流程图素材。
- 若后续你决定加入更长 few-shot 或多阶段提示词，请在本文件“第5节”增补版本号与变更日志。
