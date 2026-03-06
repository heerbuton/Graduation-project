# 伯牙解谱 (Boyajiepu) 后续待办事项 (TODO)

系统的前后端骨架与核心渲染管线已经全部搭建完毕。为了让系统真正具备 AI 推理能力，你需要完成以下 4 个核心模块的**真实逻辑替换**。

所有的替换工作都在 `backend/pipeline/` 目录下进行。

## 1. 模块 A：接入你自己的 YOLO 视觉检测权重
**目标文件**：`backend/pipeline/cv_module.py`

- [ ] **拷贝权重**：将你训练好的 YOLO 权重文件（如 `best.pt`）放入 `backend/` 目录下（或其子目录，如 `backend/assets/`）。
- [ ] **修改代码**：
  - 在 `cv_module.py` 顶部导入 `from ultralytics import YOLO`。
  - 实例化模型：`model = YOLO('你的权重路径.pt')`。
  - 移除我写的 `Mock 返回` 的硬编码字典。
  - 在 `detect_components` 函数内调用模型：`results = model(image_path)`。
  - 将 `results` 中的边界框 (`boxes.xyxy`) 和分类 (`boxes.cls`) 转换为要求的返回格式：`[{"class": "大", "bbox": [x1, y1, x2, y2], "conf": 0.95}, ...]`。

## 2. 模块 B：实现真实的空间拓扑算法
**目标文件**：`backend/pipeline/topology_module.py`

- [ ] **设计算法**：接收 YOLO 输出的一堆独立部件框（如："大", "九", "勾" 的坐标）。
- [ ] **合并字集**：通过计算边框中心点的欧氏距离、或是重合度，将属于同一个古琴减字谱汉字的偏旁部首们聚类到一起。
- [ ] **修改代码**：
  - 移除我写的 `parsed["group_1"]` 假数据。
  - 编写你的空间聚类循环逻辑，并输出结构化的合并结果字典。

## 3. 模块 C：接入大模型 (LLM) 打谱推理
**目标文件**：`backend/pipeline/llm_module.py`

- [ ] **准备 API Key**：如果你使用 ChatGPT、智谱 GLM、通义千问等模型，请获取它们的 API Key。
- [ ] **组装 Prompt**：根据输入的 `topology_json` 结构，构建一段系统提示词。比如：“你是一个古琴打谱专家。现在有以下减字谱识别结果 {topology_json}，请结合正调法推断它们的简谱音高及时值。”
- [ ] **修改代码**：
  - 在 `infer_pitch_duration` 中使用 `requests` 或官方 SDK 调用 LLM 的对话接口。
  - 解析大模型返回的文本，强烈建议通过 Prompt 设置让大模型强制返回 JSON 格式，以便反序列化为后端生成器要求的规范列表（支持小节切换、八度及四项技法）：`[{"pitch": "1", "octave": "4", "duration": "4", "action": "勾", "string": "一", "position": "九", "finger": "大", "new_measure": true}]`。

## 4. 模块 D：MusicXML 渲染及验证 (可选)
**目标文件**：`backend/pipeline/musicxml_encoder.py`

- [ ] **准备 Schema**：(可选) 将 MusicXML 3.1 DTD 或 XSD 文件保存至后端。
- [ ] **加强校验 (修改代码)**：
  - 安装 lxml 库：`pip install lxml`
  - 使用 `lxml.etree` 进行严格的验证。
  - 检查生成的 MusicXML 中歌词 (lyric) 等字段是否合法组合了减字谱数据。

---

## ✅ 最终端到端测试
当上述 4 步（或至少前 3 步）完成后：
1. 重启后端的 Flask 服务。
2. 确保前端的 Vite 服务在运行。
3. 打开浏览器，这一次**不要**点击右上角的 Mock 按钮。
4. 直接拖拽一张你自己的古琴谱图片到左侧虚线框内，点击**[开始端到端解析]**。
5. 观察系统从真正运行 YOLO 到 LLM 推理的完整流水线，并最终在右侧渲染出真正的解谱乐谱！
