# 伯牙解谱 (Boyajiepu) 端到端 Web 演示系统原型

本项目是《基于 LLM 的古琴谱自动翻译算法及系统开发》（项目名：伯牙解谱）的前后端分离 Web 演示系统原型。
系统采用 Vue 3 + Tailwind CSS 作为前端展示，Flask 作为后端核心流水线驱动。旨在实现古琴谱扫描件经过视觉感知（YOLO）、空间拓扑解析、大型语言模型（LLM）推理打谱后，封装成包含 4 行 `<lyric>` 特殊减字谱元数据的 MusicXML 数据，并最终在 Web 端呈现为双层古琴翻译乐谱。

---

## 目录结构
```text
f:/AIcharacter/End/
├── backend/                  # Flask 后端服务
│   ├── app.py                # 主入口应用 (防崩路由与Mock)
│   ├── requirements.txt      # 后端依赖
│   ├── static/uploads/       # 识别图片防乱码落盘区 
│   ├── pipeline/             # 4大核心模块 (脚手架/Stub/Mock)
│   │   ├── cv_module.py      # 模块A: YOLO视觉感知 (可替换为你自己的 weights)
│   │   ├── topology_module.py# 模块B: 空间拓扑解析
│   │   ├── llm_module.py     # 模块C: LLM大模型打谱调用
│   │   └── musicxml_encoder.py# 模块D: MusicXML 生成层 
├── frontend/                 # Vue 3 前端服务
│   ├── package.json
│   ├── index.html
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── vite.config.js
│   └── src/
│       ├── main.js
│       ├── index.css
│       ├── App.vue           # 左侧上传与原图Canvas渲染，右侧渲染器挂载
│       └── components/
│           └── MusicXmlScoreRenderer.vue # ★核心 MusicXML 解析器与基于Flex的乐谱渲染组件
└── README.md
```

—

## 预留自行接入说明 (针对您自有的资源)

本架构设计了标准的数据握手层。请在相应的脚手架模块中加载您的内部资源：

1. **YOLO `best.pt` 权重文件**:
   - 打开 `backend/pipeline/cv_module.py`
   - 将 `detect_components` 方法的核心逻辑替换为 `from ultralytics import YOLO` 并利用其进行图像推理。
2. **MusicXML Schema 验证**:
   - 打开 `backend/pipeline/musicxml_encoder.py`
   - 当生成 XML 字符串后，可引入 `lxml` 等库，加载标准 MusicXML 规范文件进行严谨的模式校验。

---

## 快速上手与运行指南

### 1. 启动后端服务 (Flask)

打开一个终端窗口，确保您已安装 Python 3.8+环境，执行以下命令：

```powershell
cd f:/AIcharacter/End/backend

# 强力建议创建并激活虚拟环境 (可选)
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行 Flask
python app.py
```
*启动成功后将在 `http://0.0.0.0:5000` 提供 API 服务。*

### 2. 启动前端服务 (Vue / Vite)

打开另一个终端窗口，要求 Node.js 16+：

```powershell
cd f:/AIcharacter/End/frontend

# 安装依赖项 (如果您刚刚才执行过 npm install，可跳过此步)
npm install

# 启动开发服务器
npm run dev
```

*打开终端中显示的网址（通常为 `http://localhost:3000` 或 `http://localhost:5173`）即可预览应用。*

---

## 联调测试方法

1. **界面交互**: 浏览器打开网页后，可看见充满极客设计的深色顶栏与左右分栏页面。
2. **测试 Mock 通路**: 页面右上角点击 `[加载 Mock XML 并渲染]`，无需上传图片，前端将直接请求 `/api/mock_pipeline` 内置的丰富 MusicXML 测试片段，您将在右侧全貌预览到：
   - 绿色区域提示 "MusicXML 解析成功"
   - 基于 Vue 框架原生构筑的水平排版标准简谱。完美支持**高/低音指示点**、**增时线**（二分音符 `-` 占位）、**减时线**（八分/十六分音符底边下单/双下划线）、**小节线**，以及完美垂直对齐在每个音符下方的**四部分古琴减字信息**（右手技法、弦序、徽位、左手技法）。
   - *(可查看 `walkthroughs/images/final_rendered_result.png` 了解最新排版效果)*
3. **测试带中文的图片上传通路**: 
   - 准备一张任意图片，重命名包含中文（例如: `测试图片_古琴.jpg`）以验证防崩落盘体系。
   - 拖拽或点击左侧上传框进行上传，并点击 `[开始端到端解析]`。
   - 观察动态 Status 流水线 (`特征提取` -> `拓扑解析` -> `LLM` -> `XML`)。
   - 成功后，左侧出现原图并在上面用红框绘制了 Mock 的 YOLO 边框与置信度。右侧同样生成 MusicXML 解析后的乐谱渲染块。
