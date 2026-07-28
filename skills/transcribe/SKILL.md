---
name: transcribe
description: 视频/音频转文字。用 Whisper 将视频或音频文件转写为带时间戳的文字。支持模型选择、音频拆分、自动代理。Apple Silicon 上使用 GPU 加速。
version: 0.1.0
---

# 视频/音频转文字

将视频或音频文件转写为带时间戳的文字。在 Apple Silicon Mac 上使用 GPU 加速。

## 脚本位置

Skill 自带的 Python 脚本：

```bash
python skills/transcribe/scripts/transcribe.py <文件路径> [参数]
```

> 脚本路径相对于插件根目录。如果插件安装在 `~/.claude/plugins/transcribe-plugin/`，则完整路径为 `~/.claude/plugins/transcribe-plugin/skills/transcribe/scripts/transcribe.py`。

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 模型: tiny / small / medium / large-v3 | medium |
| `-l, --language` | 语言: zh / en / auto / ja / ko ... | zh |
| `-o, --output` | 输出文件路径 | 输入文件名.txt |
| `--split N` | 平均拆分为 N 份分别转写 | 不拆分 |
| `--split-duration MIN` | 按分钟数拆分 | 不拆分 |
| `--backend` | auto / mlx / faster | auto |
| `--keep-wav` | 保留提取的 WAV | 删 |
| `--keep-chunks` | 保留拆分片段 | 删 |
| `--proxy` | HTTP 代理地址 | 不走代理 |

## 工作流程

1. 如果是视频 → ffmpeg 提取 WAV 音频（16kHz 单声道）
2. 如果指定了 `--split` / `--split-duration` → 拆分音频
3. 加载 Whisper 模型（首次自动下载，之后走缓存）
4. 逐段转写，输出带时间戳的文本
5. 如果有拆分 → 合并所有段，清理临时文件

## 示例

```bash
# 基础转写
python scripts/transcribe.py meeting.mp4

# small 模型 + 英文
python scripts/transcribe.py interview.wav -m small -l en

# 超长录音切 6 段
python scripts/transcribe.py long_lecture.wav --split 6

# 每 20 分钟一段
python scripts/transcribe.py lecture.wav --split-duration 20

# large-v3 高精度
python scripts/transcribe.py important.mp4 -m large-v3 -o important.txt
```

## 速度参考（Apple Silicon）

| 模型 | 实时倍数 | 2 小时录音耗时 |
|------|---------|---------------|
| tiny | 50-80x | ~2 分钟 |
| small | 30-50x | ~3 分钟 |
| medium | 15-25x | ~6 分钟 |
| large-v3 | 5-10x | ~15 分钟 |

## 存储预测

### 模型缓存（仅下载一次）

| 模型 | 大小 |
|------|------|
| tiny | ~150 MB |
| small | ~500 MB |
| medium | ~1.5 GB |
| large-v3 | ~3 GB |

4 个全下约 5.2 GB；日常 medium + small 约 2 GB。缓存路径在 `~/.cache/huggingface/hub/`。

### 单次运行临时空间

| 音频时长 | 临时 WAV 大小 |
|---------|-------------|
| 10 分钟 | ~115 MB |
| 1 小时 | ~690 MB |
| 2 小时 | ~1.4 GB |
| 8 小时 | ~5.5 GB |

> 公式：16kHz 16-bit 单声道 = ~11.5 MB/分钟。拆分不会产生额外空间（切分不复制）。转写完成后临时 WAV 自动删除。

## 执行后

1. 告诉用户转写耗时
2. 失败时根据错误类型判断：代理不通？模型下载失败？ffmpeg 未安装？
3. 不要自动读取转写结果文件（可能很长），告诉用户路径即可
