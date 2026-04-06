```mermaid
flowchart LR
    A[YOLO检测输出
class+bbox+conf] --> B[Topology聚合
group_1...group_N]
    B --> C[文本组装
紧凑JSON序列]
    C --> D[LLM打谱
notes(JSON)]
    D --> E[结构化校验
字段/顺序/枚举]
    E --> F[MusicXML编码
note/pitch/duration]
    F --> G[渲染展示
MuseScore/Web]
```
