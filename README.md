# 伯牙解谱 (Boyajiepu) 端到端 Web 演示系统原型

本项目是《基于 LLM 的古琴谱自动翻译算法及系统开发》的前后端分离原型系统。  
后端采用 Flask 组织流水线，前端采用 Vue 3 + Vite 展示结果。

当前版本同时输出并支持两种渲染数据：

- `music_xml`：标准 MusicXML
- `score_model`：前端直接渲染的结构化谱面模型（推荐主链路）

---

## 最新能力（已落地）

- 支持 `LLM JSON -> Python 规范化 -> ScoreModel -> 前端渲染`
- 支持自动分小节（4/4 时值推断 + 显式 `new_measure` 混合策略）
- 支持谱面换行显示（当前为每行 3 小节）
- 支持小节线显示
- 前端支持一键加载 `testpicture-1` 已落盘结果进行渲染检查

---

## 目录结构（关键部分）

```text
f:/AIcharacter/End/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── static/uploads/
│   └── pipeline/
│       ├── cv_module.py
│       ├── topology_module.py
│       ├── llm_module.py
│       ├── musicxml_encoder.py
│       └── score_model_transformer.py   # 新增：LLM结果 -> ScoreModel
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.vue
│       ├── components/
│       │   └── ScoreModelRenderer.vue   # 新增：ScoreModel渲染组件
│       └── utils/
│           ├── scoreModel.js            # 新增：前端fallback转换
│           └── scoreModel.test.mjs
├── scripts/
│   └── convert_llm_json_to_score_model.py  # 新增：离线转换脚本
└── test/
    ├── test_app_score_model.py
    ├── test_score_model_transformer.py
    └── test_score_model_script.py
```

---

## 流水线说明

1. 模块 A：`cv_module.py` 进行视觉检测（YOLO）
2. 模块 B：`topology_module.py` 进行空间拓扑与减字序列化
3. 模块 C：`llm_module.py` 进行打谱推理，输出 `llm_result`
4. 模块 D：双输出
   - `musicxml_encoder.py` 输出 `music_xml`
   - `score_model_transformer.py` 输出 `score_model`

---

## API 返回字段（核心）

### `POST /api/upload`

返回 `data` 中包含：

- `original_image_url`
- `yolo_boxes`
- `topology_json`
- `jianzi_sequence`
- `llm_result`
- `score_model`
- `music_xml`

### `GET /api/mock_pipeline`

同样返回上述关键字段（用于前端联调）。

---

## 本地运行

### 1. 启动后端（推荐使用当前环境下的 Python 解释器）

```powershell
cd f:/AIcharacter/End/backend
F:\anaconda\envs\pytorch\python.exe -m pip install -r requirements.txt
F:\anaconda\envs\pytorch\python.exe app.py
```

后端 API 与静态资源服务默认运行在：`http://127.0.0.1:5000`

### 2. 启动前端（推荐绑定 IPv4 物理网卡以防止连接拒绝）

由于最新的 Node.js 策略有时会将 localhost 强制路由为 IPv6(::1)，建议直接将 host 显式绑定在 127.0.0.1 上：

```powershell
cd f:/AIcharacter/End/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
# 如果脚本包装异常，也可直接执行: 
# .\node_modules\.bin\vite.cmd --host 127.0.0.1 --port 5173
```

前端网页界面访问地址：**`http://127.0.0.1:5173/`**

---

## 体验与查看方式（基于最新 Accordion 手风琴卡片 UI）

### 方式 A：无缝体验海量数据的流水线展开（强烈推荐）

当前端网页加载完成后：
1. 直接点击顶栏的 **`Demo: 加载测试图`** 按钮。
2. 左侧控制台将调取内部已恢复好且跑通的 `testpicture-1.jpg_result.json`（内含 300 多个解析框）。
3. **视觉特征抽取 -> 拓扑结构序列化 -> AI 推理 -> XML 编码** 这四大核心步骤对应的控制卡片会自动“像手风琴一样”随着进度节奏依次点亮、展开、并最终渲染古琴电子版乐谱，效果极其丝滑爽快。

### 方式 B：真实上传端到端解析

1. 拖拽或点击上传本地源图片。
2. 点击 **`启动 AI 打谱引擎`**，系统会将图片传回本地的 Flask 引擎实时执行深度学习推理，然后动态推演状态机。

---

## 离线转换脚本（LLM JSON -> ScoreModel）

```powershell
cd f:/AIcharacter/End
F:\anaconda\envs\pytorch\python.exe scripts/convert_llm_json_to_score_model.py --input <input_json> --output <output_json>
```

输入支持：

- 直接数组：`[ {...}, {...} ]`
- 对象包裹：`{"llm_result":[...]}` 或 `{"notes":[...]}`

---

## 测试与构建

```powershell
cd f:/AIcharacter/End
F:\anaconda\envs\pytorch\python.exe -m unittest discover -s test -p "test_score_model_transformer.py"
F:\anaconda\envs\pytorch\python.exe -m unittest discover -s test -p "test_app_score_model.py"
F:\anaconda\envs\pytorch\python.exe -m unittest discover -s test -p "test_score_model_script.py"

cd f:/AIcharacter/End/frontend
node --test src/utils/scoreModel.test.mjs
npm run build
```
