# 伯牙解谱：古琴减字谱自动识别与打谱系统

本仓库是毕业设计项目《基于 LLM 的古琴谱自动翻译算法及系统开发》的最终版代码。项目已经完成，后续不再进行常规功能更新；仓库保留为答辩、复现和展示用的终稿版本。

系统采用前后端分离架构：后端用 Flask 串联 YOLO 视觉检测、拓扑聚合、LLM 语义打谱和 MusicXML/ScoreModel 输出；前端用 Vue 3 + Vite 展示上传、识别流程、人工修正和最终谱面渲染结果。

---

## 项目状态

- 当前状态：终稿归档版
- 更新策略：不再做常规迭代，仅保留必要的运行说明和代码整理
- 论文材料：本地已整理到 `论文支持/`，该目录不会上传到 GitHub
- 远端仓库：仅保留项目运行所需代码、模型入口、测试脚本和说明文档

---

## 核心能力

- 上传古琴减字谱图片并执行端到端识别流程
- 使用 YOLO 权重进行减字谱部件检测
- 支持 `Yolo/README.md` 中记录的多模型 SAHI 切片召回优先方案
- 对检测框进行拓扑排序、空间聚合和减字序列化
- 调用 LLM 将减字结构转换为结构化打谱结果
- 同时输出 `music_xml` 和前端可直接渲染的 `score_model`
- 前端支持流程态展示、局部人工修正、重新推理和最终谱面渲染
- 测试脚本可输出红框标注的检测结果图片，便于检查识别流程

---

## 目录结构

```text
f:/AIcharacter/End/
├── backend/
│   ├── app.py
│   ├── best.pt
│   ├── requirements.txt
│   └── pipeline/
│       ├── cv_module.py
│       ├── topology_module.py
│       ├── llm_module.py
│       ├── musicxml_encoder.py
│       └── score_model_transformer.py
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.vue
│       ├── components/
│       └── utils/
├── Yolo/
│   ├── README.md
│   ├── reports/
│   ├── scripts/
│   └── weights/              # 本地权重目录，已忽略，不上传
├── scripts/
│   ├── convert_llm_json_to_score_model.py
│   ├── git-sync.ps1
│   └── Use.md
├── test/
│   ├── run_backend_pipeline_test.py
│   └── *.py / 测试输出样例
├── walkthroughs/
├── start_all.bat
├── todo.md
└── 论文支持/                 # 本地论文支撑材料，已忽略，不上传
```

`论文支持/` 中归档了原来的 `周日志/`、`paper_assets/`、`thesis_materials_*`、`相关文档/` 等论文支撑材料。它们不再作为仓库内容上传，避免 GitHub 仓库继续膨胀。

---

## 后端流水线

1. `cv_module.py`：执行 YOLO/SAHI 视觉检测，输出候选框
2. `topology_module.py`：根据空间位置聚合减字组件并生成序列
3. `llm_module.py`：调用 LLM 进行语义打谱推理
4. `musicxml_encoder.py`：生成 MusicXML
5. `score_model_transformer.py`：生成前端渲染用的 ScoreModel

主要接口：

- `POST /api/upload`：上传图片并执行完整流水线
- `GET /api/mock_pipeline`：加载测试数据用于前端联调
- `POST /api/reflow`：基于修正后的结构重新生成谱面结果

---

## 本地运行

### 一键启动

在项目根目录运行：

```bat
start_all.bat
```

脚本会分别启动后端 Flask 服务和前端 Vite 服务。

### 手动启动后端

```powershell
cd f:/AIcharacter/End/backend
F:\anaconda\envs\pytorch\python.exe -m pip install -r requirements.txt
F:\anaconda\envs\pytorch\python.exe app.py
```

后端默认地址：

```text
http://127.0.0.1:5000
```

### 手动启动前端

```powershell
cd f:/AIcharacter/End/frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

前端访问地址：

```text
http://127.0.0.1:5173/
```

---

## YOLO 权重说明

- `backend/best.pt`：后端默认可用的 YOLO 权重入口
- `Yolo/weights/`：本地多模型权重目录，已加入 `.gitignore`
- `Yolo/README.md`：记录最终测试使用的 YOLO 召回优先方案和脚本说明

如果需要使用多模型 SAHI 融合方案，请按 `Yolo/README.md` 准备本地权重；如果只做基础演示，后端会优先使用当前可用的默认权重。

---

## 测试

后端流水线测试：

```powershell
cd f:/AIcharacter/End
F:\anaconda\envs\pytorch\python.exe test/run_backend_pipeline_test.py
```

该脚本会使用测试图片跑通识别流程，并在 `test/pipeline_test_outputs/` 输出结果图片。检测框使用红色矩形标注，不显示类别名。

ScoreModel 与接口测试：

```powershell
cd f:/AIcharacter/End
F:\anaconda\envs\pytorch\python.exe -m unittest discover -s test -p "test_score_model_transformer.py"
F:\anaconda\envs\pytorch\python.exe -m unittest discover -s test -p "test_app_score_model.py"
F:\anaconda\envs\pytorch\python.exe -m unittest discover -s test -p "test_score_model_script.py"
```

前端构建：

```powershell
cd f:/AIcharacter/End/frontend
npm run build
```

---

## Git 上传说明

本项目保留了一键提交脚本：

```powershell
git sync
```

运行后会提示输入提交说明，回车后自动暂存、提交并推送当前分支。脚本不会自动同步远端内容；如果确实需要先拉取远端更新，可使用：

```powershell
git sync -PullFirst
```

更多说明见 `scripts/Use.md`。

当前 `.gitignore` 已排除：

- `论文支持/`
- `周日志/`
- `paper_assets/`
- `相关文档/`
- `thesis_materials_*/`
- `Yolo/weights/`
- 前后端依赖、运行缓存、上传文件和临时日志

---

## 终稿说明

本仓库现在聚焦于系统本体代码和可复现运行流程。论文写作支撑材料仍保留在本地 `论文支持/`，但不再进入 GitHub 仓库；云端历史中已经存在的论文支撑目录会在本次终稿提交后从远端仓库移除。
