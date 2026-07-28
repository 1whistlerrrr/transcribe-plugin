# Transcribe

视频/音频 → 文字转写 Claude Code 插件。在 Apple Silicon Mac 上使用 GPU 加速，2 小时录音 tiny 模型约 2 分钟。

## 安装

### 1. 安装系统依赖

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> `requirements.txt` 只需要两个包：`mlx-whisper`（Apple GPU 加速）和 `faster-whisper`（CPU 备选）。
> 如果你只在 Mac 上用，只装 `mlx-whisper` 就够了。

### 3. 安装插件

**方式 A：插件市场（推荐）**

```
/plugin marketplace add 1whistlerrrr/transcribe-plugin
/plugin install transcribe@transcribe-plugin
```

**方式 B：手动安装**

```bash
git clone https://github.com/1whistlerrrr/transcribe-plugin.git ~/.claude/plugins/transcribe-plugin
```

然后在 Claude Code 中 `/reload-plugins`。

## 模型

| 模型 | 大小 | 速度（2 小时录音） | 准确度 |
|------|------|-------------------|--------|
| tiny | ~150MB | ~2 分钟 | 一般 |
| small | ~500MB | ~3 分钟 | 还行 |
| medium | ~1.5GB | ~6 分钟 | 较好 |
| large-v3 | ~3GB | ~15 分钟 | 最佳 |

首次使用会自动下载模型并缓存到 `~/.cache/huggingface/hub/`。

## 功能

- ✅ 视频自动提取音频（ffmpeg）
- ✅ Apple Silicon GPU 加速（mlx-whisper）
- ✅ CPU 备选后端（faster-whisper）
- ✅ 超长音频拆分转写（`--split` / `--split-duration`）
- ✅ 自动检测 Clash 代理下载模型
- ✅ 临时文件自动清理

## License

MIT
