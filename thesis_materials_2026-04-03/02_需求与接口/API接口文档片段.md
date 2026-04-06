# API 接口文档片段（核心）

> 说明：以下接口定义基于 `backend/app.py` 实际实现，可直接用于论文“接口设计”小节。

## 1. `POST /api/upload`

### 功能
上传用户图片并执行完整流水线：
`YOLO -> topology -> jianzi_sequence -> LLM -> score_model -> MusicXML`

### 请求

- Content-Type: `multipart/form-data`
- 字段：
  - `file`：图片文件（二进制）

### 成功响应（`200`）

```json
{
  "status": "success",
  "data": {
    "original_image_url": "/static/uploads/<filename>",
    "yolo_boxes": [],
    "topology_json": {},
    "jianzi_sequence": [],
    "llm_result": [],
    "score_model": {},
    "music_xml": "<score-partwise>...</score-partwise>"
  }
}
```

### 失败响应

- `400`：缺失文件字段、空文件、非法图片
- `500`：保存失败或流水线执行失败

---

## 2. `POST /api/reflow_from_topology`

### 功能
基于前端修正后的 `topology_json`，仅重跑后续链路，不重复 YOLO 检测。

### 请求

```json
{
  "topology_json": {
    "group_1": {
      "right_fingering": "勾",
      "left_fingering": "",
      "left_finger": "大",
      "hui": "七",
      "xian": "一",
      "__deleted": false
    }
  }
}
```

### 成功响应（`200`）

```json
{
  "status": "success",
  "data": {
    "topology_json": {},
    "jianzi_sequence": [],
    "llm_result": [],
    "score_model": {},
    "music_xml": "<score-partwise>...</score-partwise>"
  }
}
```

### 失败响应

- `400`：`topology_json` 为空或全部被删除
- `500`：后续链路重跑失败

---

## 3. `GET /api/run_testpicture_pipeline`

### 功能
固定样例 `testpicture-1.jpg` 端到端跑通，用于联调/回归。

### 响应结构
同 `POST /api/upload` 的成功响应。

---

## 4. `GET /api/mock_pipeline`

### 功能
返回 Mock 结果，供前端渲染开发。

### 特点

- 包含示例 `llm_result`
- 包含示例 `score_model`
- 包含示例 `music_xml`

---

## 5. 静态资源接口

### `GET /static/uploads/<filename>`

用于访问上传后的原图或结果图，前端通过 `original_image_url` 直接加载。

---

## 6. 核心数据对象（论文中建议单独列字段）

1. `yolo_boxes`：检测框集合，元素含 `class/bbox/confidence`
2. `topology_json`：按 `group_x` 组织的聚合语义对象
3. `jianzi_sequence`：供 LLM 输入的轻量序列
4. `llm_result`：打谱推理结果（音高、八度、时值等）
5. `score_model`：前端渲染模型
6. `music_xml`：标准乐谱交换格式文本

---

## 7. 论文引用建议句

系统通过 `upload` 与 `reflow_from_topology` 两个核心接口分别支撑“全流程初次推理”与“人工修正后局部重算”，从而在保证识别质量的同时兼顾交互效率。

