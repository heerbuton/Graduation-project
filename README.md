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

### 1. 启动后端（推荐使用你当前环境）

```powershell
cd f:/AIcharacter/End/backend
F:\anaconda\envs\pytorch\python.exe -m pip install -r requirements.txt
F:\anaconda\envs\pytorch\python.exe app.py
```

后端默认地址：`http://127.0.0.1:5000`

### 2. 启动前端

```powershell
cd f:/AIcharacter/End/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

前端访问地址：`http://127.0.0.1:5173/index.html`

---

## 前端查看方式

### 方式 A：直接看 `testpicture-1` 结果（推荐）

前端顶栏点击：`加载 testpicture-1 结果`  
会读取：`backend/static/uploads/testpicture-1.jpg_result.json`

### 方式 B：Mock 联调

前端顶栏点击：`加载 Mock XML 并渲染`

### 方式 C：上传图片走全链路

点击上传并执行 `开始端到端解析`

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
