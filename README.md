# 👗 AI 虚拟试衣间

基于阿里云百炼 OutfitAnyone 模型的 AI 虚拟试衣应用，支持半身照自动补全为全身照、智能试衣、AI 换脸和时尚人格画像分析。

## ✨ 功能特性

- **🦵 半身照自动补全**：基于通义万相图生图技术，将半身照智能扩展为全身照，保持原人物发型、五官、体型等特征
- **👔 AI 智能试衣**：上传人物照片和衣服图片，一键生成试穿效果
- **🎭 时尚人格画像**：根据穿搭风格生成有趣的时尚人格分析（基于 2024-2025 网络热梗）
- **🔄 AI 换脸工具**：基于 TinyFace（MagicMirror 核心）的独立换脸功能
- **💰 低成本**：新用户免费 400 张额度，试衣仅需 0.20~0.50 元/张

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/ai-tryon-tool.git
cd ai-tryon-tool
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，填入你的阿里云百炼 API Key
```

编辑 `.env` 文件：

```env
DASHSCOPE_API_KEY=sk-your-api-key-here
```

> **获取 API Key**：访问 [阿里云百炼控制台](https://bailian.console.aliyun.com) → API-KEY 管理 → 创建新的 API Key

### 5. 下载 AI 换脸模型（可选）

如果需要使用 AI 换脸功能，需要下载以下模型文件到 `models_cache` 目录：

| 模型文件 | 用途 | 下载地址 |
|---------|------|---------|
| `scrfd_2.5g.onnx` | 人脸检测 | [下载](https://github.com/facefusion/facefusion-assets/releases) |
| `arcface_w600k_r50.onnx` | 人脸识别 | [下载](https://github.com/facefusion/facefusion-assets/releases) |
| `inswapper_128_fp16.onnx` | 换脸核心 | [下载](https://github.com/facefusion/facefusion-assets/releases) |
| `gfpgan_1.4.onnx` | 人脸增强 | [下载](https://github.com/facefusion/facefusion-assets/releases) |

模型会自动下载，如果下载失败可手动放置到 `models_cache` 文件夹。

### 6. 启动应用

```bash
streamlit run app.py
```

访问 http://localhost:8501 开始使用。

## 📖 使用指南

### 试衣流程

1. **输入 API Key**：在左侧边栏输入阿里云百炼 API Key（或配置 `.env` 文件）
2. **选择模型**：基础版（0.20 元/张，~15秒）或 Plus 版（0.50 元/张，~30秒，效果更好）
3. **开启半身照补全**（推荐）：自动将半身照扩展为全身照
4. **上传人物照片**：支持半身照或全身照
5. **上传衣服图片**：上装或下装至少一件（可直接截图保存服装图片上传）
6. **点击开始试穿**：等待 AI 生成结果

### AI 换脸工具

页面下方的独立功能：
- **A 照片**：提供脸的照片（你想用谁的脸）
- **B 照片**：被换脸的照片（你想把脸换到哪张照片上）
- 点击「开始换脸」即可

### 时尚人格画像

试衣完成后，系统会根据你上传的服装颜色自动生成有趣的时尚人格分析，包含：
- 时尚人格类型（如 G-O-R-P 山系户外研究猿、B-R-A-T 夏日美学先锋等）
- 人格特质、时尚宣言、口头禅
- AI 穿搭分析师点评
- 最佳搭配建议和时尚警告

## 💰 费用说明

| 功能 | 费用 | 说明 |
|------|------|------|
| 图生图全身照 | ~0.04 元/张 | 半身照补全为全身照 |
| AI 试衣-基础版 | 0.20 元/张 | ~15秒，标准效果 |
| AI 试衣-Plus版 | 0.50 元/张 | ~30秒，效果更好 |
| AI 换脸 | 免费 | 本地运行，无需 API |

> 新用户可免费获得 400 张试衣额度！

## 🎭 时尚人格类型

| 类型 | 名称 | 特点 |
|------|------|------|
| **G-O-R-P** | Gorpcore 山系户外研究猿 | 冲锋衣是皮肤，始祖鸟是信仰 |
| **B-R-A-T** | Brat 夏日美学先锋 | 荧光绿本命，派对永远不散场 |
| **C-H-I-L-L** | Chill Guy 松弛感大师 | 灰色毛衣战袍，双手插兜走天下 |
| **L-O-S-E-R** | Loser Core 反骨美学 | 反精致是态度，邋遢是艺术 |
| **M-O-B-W** | Mob Wife 大佬女人 | 豹纹基本款，皮草是日常 |
| **C-O-Q-U-E** | Coquette 蝴蝶结精 | 蝴蝶结是本体，粉色是空气 |
| **O-L-D-M** | Old Money 老钱贵族 | Logo 是禁忌，质感是唯一标准 |
| **Y-2-K-R** | Y2K 复古千禧女王 | 低腰裤信仰，2000年从未离开 |
| **P-R-E-P-P** | Preppy 预科风学术王子 | 格子衬衫校服，图书馆是 T 台 |
| **N-O-R-M-C** | Normcore 基础款传奇 | 白 T 是礼服，普通是最高级 |

## 🛠️ 技术栈

- **前端**：Streamlit
- **AI 试衣**：阿里云百炼 OutfitAnyone 模型
- **图生图**：通义万相 wan2.6-image 模型
- **AI 换脸**：TinyFace（MagicMirror 核心，基于 InsightFace）
- **图像处理**：Pillow、OpenCV

## 📁 项目结构

```
ai-tryon-tool/
├── app.py                 # 主应用文件
├── requirements.txt       # Python 依赖
├── .env.example          # 环境变量示例（复制为 .env 后配置）
├── .gitignore            # Git 忽略文件（已包含 .env 和 models_cache）
├── models_cache/         # AI 换脸模型缓存目录
│   ├── scrfd_2.5g.onnx
│   ├── arcface_w600k_r50.onnx
│   ├── inswapper_128_fp16.onnx
│   └── gfpgan_1.4.onnx
└── README.md             # 项目说明
```

## ⚠️ 隐私与安全

- **API Key 保护**：请勿将 `.env` 文件提交到 GitHub，已添加 `.gitignore` 保护
- **图片处理**：所有图片处理均在本地完成，不会上传到第三方服务器（除阿里云百炼 API 调用外）
- **换脸模型**：模型文件较大（约 300MB），首次使用会自动下载到 `models_cache` 目录

## 🔧 常见问题

### Q: 模型下载失败怎么办？
A: 手动下载模型文件并放置到 `models_cache` 目录，参考上方「下载 AI 换脸模型」部分。

### Q: 为什么换脸结果不像？
A: 请确保两张照片都包含清晰、正面的人脸，光线充足，无遮挡。

### Q: 半身照补全后人物特征变了？
A: 图生图技术已优化，会尽量保持原人物特征。如效果不佳，建议直接上传全身照。

### Q: 可以商用吗？
A: 请遵守阿里云百炼的使用条款，API 调用产生的费用需自行承担。

## 📄 开源协议

MIT License

## 🙏 致谢

- [阿里云百炼](https://bailian.console.aliyun.com) - AI 试衣 API
- [MagicMirror](https://github.com/haofanwang/MagicMirror) - TinyFace 换脸技术
- [Streamlit](https://streamlit.io) - Web 应用框架
