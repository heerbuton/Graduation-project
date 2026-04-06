# LLM 选型与配置记录（音位表增强版）

- 记录时间：2026-04-04 20:54:21
- 模型：`qwen3.5-plus`
- API 地址：`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
- Temperature：`0.0`
- enable_thinking：`False`
- 超时（秒）：`180.0`
- response_format：`json_object`
- Top-p：代码中未显式设置（平台默认值）

## 额外配置（与论文相关）
- 音位表来源：`相关文档/音位表.xlsx`
- 映射文件：`backend/pipeline/qin_tone_table.json`
- 注入策略：在发送给 LLM 的每个 group 中增加 `tone_table_*` 字段
- 约束策略：`tone_table_hit=true` 时提示词要求优先使用映射音高八度

## 请求参数快照
```json
{
  "model": "qwen3.5-plus",
  "temperature": 0.0,
  "enable_thinking": false,
  "response_format": {
    "type": "json_object"
  }
}
```
