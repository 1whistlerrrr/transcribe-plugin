# Transcribe

视频/音频 → 文字转写工具。基于 Whisper，在 Apple Silicon Mac 上使用 GPU 加速（mlx-whisper）。

2 小时录音，tiny 模型约 2 分钟出结果。

---

## 两种使用方式

### 方式 1：作为 Claude Code 插件（`/transcribe`）

在 Claude Code 中直接用自然语言驱动转写，Claude 帮你处理参数和排错。

**安装**

```bash
# 先在 Claude Code 中注册插件市场
/plugin marketplace add 1whistlerrrr/transcribe-plugin

# 再安装插件
/plugin install transcribe@transcribe-marketplace
```

> 原理就是 `git clone` 这个 repo，Claude Code 读取 `.claude-plugin/marketplace.json` 知道有哪些插件，然后加载 `skills/transcribe/SKILL.md`。不需要注册任何账号或 API key。

**使用**

```
/transcribe meeting.mp4 -m small
/transcribe long_interview.wav --split 4 -m medium
/transcribe podcast.mp3 -l en -m tiny
```

---

### 方式 2：命令行直接运行

不装插件也能用，直接跑 Python 脚本。

```bash
# 克隆仓库
git clone https://github.com/1whistlerrrr/transcribe-plugin.git
cd transcribe-plugin

# 安装依赖
pip install -r requirements.txt

# 直接跑
python skills/transcribe/scripts/transcribe.py video.mp4
python skills/transcribe/scripts/transcribe.py audio.wav -m large-v3 -l en
python skills/transcribe/scripts/transcribe.py long_lecture.wav --split 6
```

---

## 安装系统依赖

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

Python 依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 只声明了 `mlx-whisper`（Apple GPU）和 `faster-whisper`（CPU 备选），装其中一个即可。

---

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<文件>` | 输入文件（视频或音频） | 必填 |
| `-m, --model` | 模型: tiny / small / medium / large-v3 | medium |
| `-l, --language` | 语言: zh / en / auto / ja / ko ... | zh |
| `-o, --output` | 输出文件路径 | 输入文件名.txt |
| `--split N` | 平均拆分为 N 份分别转写后合并 | 不拆分 |
| `--split-duration MIN` | 按分钟数拆分后分别转写 | 不拆分 |
| `--backend` | 推理后端: auto / mlx / faster | auto |
| `--keep-wav` | 保留提取的中间 WAV 文件 | 自动删除 |
| `--keep-chunks` | 保留拆分的中间片段 | 自动删除 |
| `--proxy` | HTTP 代理地址 | 自动检测 Clash |
| `--no-proxy` | 禁用代理，直连 HuggingFace | - |

---

## 模型选择

| 模型 | 大小 | 2 小时录音耗时 | 准确度 |
|------|------|--------------|--------|
| tiny | ~150 MB | ~2 分钟 | 一般 |
| small | ~500 MB | ~3 分钟 | 还行 |
| medium | ~1.5 GB | ~6 分钟 | 较好 |
| large-v3 | ~3 GB | ~15 分钟 | 最佳 |

首次使用会自动从 HuggingFace 下载模型，缓存到 `~/.cache/huggingface/hub/`。之后不再下载。

---

## 存储空间

| 场景 | 所需空间 |
|------|---------|
| 10 分钟录音 | ~115 MB 临时 + 模型缓存 |
| 1 小时录音 | ~700 MB 临时 + 模型缓存 |
| 2 小时录音 | ~1.4 GB 临时 + 模型缓存 |

临时 WAV 文件转写完成后自动清理。拆分（`--split`）不增加额外存储开销。

---

## 工作流程

1. 视频 → ffmpeg 提取 16kHz 单声道 WAV
2. （可选）`--split` / `--split-duration` 拆分音频
3. 加载 Whisper 模型（首次下载，之后缓存）
4. 逐段转写，输出带时间戳的文字
5. 合并、清理临时文件

---

## License

MIT
