#!/usr/bin/env python3
"""
视频/音频 → 文字转写插件

后端支持:
  - mlx-whisper   （Apple Silicon GPU / ANE 加速，M 系列 Mac 首选）
  - faster-whisper （CTranslate2 CPU 推理）
  - auto          （自动检测：Apple Silicon → mlx，其他 → faster）

模型可选: tiny / small / medium / large-v3

拆分功能:
  - --split N: 将音频平均拆成 N 份，分别转写后合并（适合超长录音）
  - --split-duration MIN: 按指定时长（分钟）拆分

网络代理:
  - 默认自动检测 Clash Verge（127.0.0.1:7890）
  - 可通过 --proxy 自定义，--no-proxy 禁用
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

# ===================== 常量 =====================

MODEL_CHOICES = ["tiny", "small", "medium", "large-v3"]
BACKEND_CHOICES = ["auto", "mlx", "faster"]
DEVICE_CHOICES = ["cpu", "cuda"]

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".wmv", ".m4v", ".ts"}
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

# Clash / Clash Verge Rev 默认代理端口（按优先级）
CLASH_DEFAULT_PORTS = ["7897", "7899", "7890"]

# ===================== 工具函数 =====================


def run_ffprobe(audio_path: str) -> dict:
    """用 ffprobe 获取音频元信息（时长、采样率等）。"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"❌ ffprobe 解析失败: {e}", file=sys.stderr)
        sys.exit(1)


def get_audio_duration(audio_path: str) -> float:
    """获取音频总时长（秒）。"""
    info = run_ffprobe(audio_path)
    duration = float(info.get("format", {}).get("duration", 0))
    if duration <= 0:
        # 有些格式 duration 在 stream 里
        for stream in info.get("streams", []):
            d = float(stream.get("duration", 0))
            if d > 0:
                duration = d
                break
    return duration


def extract_audio(video_path: str, wav_path: str, sample_rate: int = 16000) -> None:
    """用 ffmpeg 从视频中提取单声道 16kHz WAV 音频。"""
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        "-y",
        wav_path,
    ]
    print(f"🎬 提取音频: {video_path}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 提取失败:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ 音频已保存: {wav_path}")


def split_audio_evenly(audio_path: str, num_parts: int, output_dir: str) -> List[Tuple[str, float]]:
    """将音频平均拆分为 N 份，返回 [(chunk_path, offset_seconds), ...]。

    每个 chunk 的开始时间会被记录下来，以便后续合并时修正时间戳。
    """
    total_duration = get_audio_duration(audio_path)
    chunk_duration = total_duration / num_parts

    print(f"\n✂️  音频总时长: {total_duration / 60:.1f} 分钟")
    print(f"   拆分为 {num_parts} 份，每份约 {chunk_duration / 60:.1f} 分钟\n")

    chunks = []
    # ffmpeg segment 按时间拆分（segment_time 可能因为关键帧而略有偏差，
    # 用 -force_key_frames 保证精确切分点，但 WAV 没有关键帧概念，直接 segment 即可）
    for i in range(num_parts):
        start_time = i * chunk_duration
        chunk_path = os.path.join(output_dir, f"chunk_{i + 1:03d}.wav")

        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-c", "copy",         # 不重新编码
            "-y",
            chunk_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        actual_duration = get_audio_duration(chunk_path)
        chunks.append((chunk_path, start_time))
        print(f"   Chunk {i + 1}/{num_parts}: "
              f"[{_fmt_time(start_time)} -> {_fmt_time(start_time + actual_duration)}] "
              f"→ {os.path.basename(chunk_path)}")

    return chunks


def split_audio_by_duration(audio_path: str, duration_min: float, output_dir: str) -> List[Tuple[str, float]]:
    """按指定时长（分钟）拆分音频，返回 [(chunk_path, offset_seconds), ...]。"""
    total_duration = get_audio_duration(audio_path)
    chunk_sec = duration_min * 60
    num_parts = max(1, math.ceil(total_duration / chunk_sec))

    print(f"\n✂️  音频总时长: {total_duration / 60:.1f} 分钟")
    print(f"   按每段 {duration_min} 分钟拆分，共 {num_parts} 段\n")

    chunks = []
    for i in range(num_parts):
        start_time = i * chunk_sec
        chunk_path = os.path.join(output_dir, f"chunk_{i + 1:03d}.wav")

        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(chunk_sec),
            "-c", "copy",
            "-y",
            chunk_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        actual_duration = get_audio_duration(chunk_path)
        chunks.append((chunk_path, start_time))
        print(f"   Chunk {i + 1}/{num_parts}: "
              f"[{_fmt_time(start_time)} -> {_fmt_time(start_time + actual_duration)}] "
              f"→ {os.path.basename(chunk_path)}")

    return chunks


def _fmt_time(seconds: float) -> str:
    """格式化时间为 HH:MM:SS。"""
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def merge_transcripts(chunk_outputs: List[str], final_output: str) -> None:
    """合并多个 chunk 的转写结果到单个文件。chunk 内的时间戳已经是绝对时间，
    因为每个 chunk 在转写时已按 offset 做了偏移。"""
    print(f"\n📦 合并 {len(chunk_outputs)} 个转写结果...")
    with open(final_output, "w", encoding="utf-8") as out:
        for i, path in enumerate(chunk_outputs):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                out.write(content)
                if i < len(chunk_outputs) - 1:
                    out.write("\n")
                print(f"   ✅ Chunk {i + 1}/{len(chunk_outputs)}: {path}")
            except FileNotFoundError:
                print(f"   ⚠️  Chunk {i + 1} 转写结果丢失: {path}", file=sys.stderr)

    print(f"📄 最终结果: {final_output}")


def is_video(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_VIDEO_EXTS


def is_audio(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_AUDIO_EXTS


def is_apple_silicon() -> bool:
    """检测是否为 Apple Silicon Mac。"""
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        return "Apple" in result.stdout
    except Exception:
        return False


def detect_clash_proxy() -> Optional[str]:
    """自动检测 Clash 代理端口（通过实际代理请求验证）。"""
    for port in CLASH_DEFAULT_PORTS:
        try:
            result = subprocess.run(
                ["curl", "-s", "--connect-timeout", "2",
                 "--proxy", f"http://127.0.0.1:{port}",
                 "-o", "/dev/null", "-w", "%{http_code}",
                 "https://huggingface.co"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() in ("200", "302", "301", "401"):
                return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return None


def setup_proxy(proxy: Optional[str], no_proxy: bool = False) -> None:
    """设置 HTTP/HTTPS 代理环境变量（huggingface_hub / requests 走这个）。"""
    if no_proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(key, None)
        print("🚫 不使用代理")
        return

    if proxy:
        proxy_url = proxy
        print(f"🔗 使用代理: {proxy_url}")
    else:
        proxy_url = detect_clash_proxy()
        if proxy_url:
            print(f"🔗 自动检测到 Clash 代理: {proxy_url}")
        else:
            print("⚠️  未检测到代理，直连 HuggingFace（可能会慢/失败）")
            print("   💡 用 --proxy 指定代理地址，或 --no-proxy 跳过此提示")
            return

    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
    os.environ["http_proxy"] = proxy_url
    os.environ["https_proxy"] = proxy_url


# ===================== 后端抽象 =====================


class TranscriberBackend(ABC):
    """转写后端基类。"""
    name: str = ""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        model_size: str,
        language: str,
        **kwargs,
    ) -> None:
        ...


class FasterWhisperBackend(TranscriberBackend):
    """CTranslate2 后端（CPU / NVIDIA CUDA）。"""

    name = "faster-whisper"

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        model_size: str,
        language: str,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
        time_offset: float = 0.0,  # 拆分时的时间偏移
        **kwargs,
    ) -> None:
        from faster_whisper import WhisperModel

        print(f"📦 加载模型: {model_size}（{self.name} | {device}/{compute_type}）...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        print(f"🎤 开始转写（偏移: {time_offset:.0f}s）...")
        start = time.time()

        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        print(f"🌐 检测语言: {info.language}（概率: {info.language_probability:.2%}）\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                # 应用时间偏移
                abs_start = seg.start + time_offset
                abs_end = seg.end + time_offset
                m1, s1 = divmod(abs_start, 60)
                m2, s2 = divmod(abs_end, 60)
                line = f"[{int(m1):02d}:{s1:05.2f} -> {int(m2):02d}:{s2:05.2f}] {seg.text.strip()}"
                f.write(line + "\n")
                print(line)

        elapsed = time.time() - start
        print(f"\n✅ 转写完成！耗时 {int(elapsed // 60)} 分 {int(elapsed % 60)} 秒")
        print(f"📄 结果已保存: {output_path}")


class MLXWhisperBackend(TranscriberBackend):
    """Apple MLX 后端（Apple Silicon GPU + ANE 加速）。"""

    name = "mlx-whisper"

    # mlx-whisper 的 HuggingFace 模型路径
    # 注意: mlx-community 命名不统一，部分带 -mlx 后缀，部分不带
    MODEL_MAP = {
        "tiny":     "mlx-community/whisper-tiny",
        "small":    "mlx-community/whisper-small-mlx",
        "medium":   "mlx-community/whisper-medium",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    }

    def transcribe(
        self,
        audio_path: str,
        output_path: str,
        model_size: str,
        language: str,
        time_offset: float = 0.0,  # 拆分时的时间偏移
        **kwargs,
    ) -> None:
        import mlx_whisper

        model_path = self.MODEL_MAP.get(model_size, model_size)
        print(f"📦 加载模型: {model_size}（{self.name} | Apple GPU/ANE | {model_path}）...")

        lang_arg = language if language and language != "auto" else None
        print(f"🎤 开始转写（偏移: {time_offset:.0f}s，语言: {lang_arg or '自动检测'}）...")
        start = time.time()

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_path,
            language=lang_arg,
            verbose=True,   # 显示进度条
            **kwargs,
        )

        detected = result.get("language", "?")
        print(f"\n🌐 检测语言: {detected}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in result.get("segments", []):
                abs_start = seg.get("start", 0) + time_offset
                abs_end = seg.get("end", 0) + time_offset
                text = seg.get("text", "").strip()
                m1, s1 = divmod(abs_start, 60)
                m2, s2 = divmod(abs_end, 60)
                line = f"[{int(m1):02d}:{s1:05.2f} -> {int(m2):02d}:{s2:05.2f}] {text}"
                f.write(line + "\n")
                print(line)

        elapsed = time.time() - start
        print(f"\n✅ 转写完成！耗时 {int(elapsed // 60)} 分 {int(elapsed % 60)} 秒")
        print(f"📄 结果已保存: {output_path}")


# ===================== 后端工厂 =====================


def detect_backend(backend_choice: str) -> TranscriberBackend:
    """根据用户选择和环境检测，返回合适的后端。"""
    apple_silicon = is_apple_silicon()

    if backend_choice == "mlx":
        if not apple_silicon:
            print("⚠️  非 Apple Silicon 设备，mlx-whisper 可能不能运行，继续尝试…")
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            print("❌ mlx-whisper 未安装。安装方法: pip install mlx-whisper", file=sys.stderr)
            sys.exit(1)
        print(f"{'🍎' if apple_silicon else '⚠️'}  使用 mlx-whisper（Apple GPU/ANE 加速）")
        return MLXWhisperBackend()

    if backend_choice == "faster":
        print("💻 使用 faster-whisper（CTranslate2）")
        return FasterWhisperBackend()

    # auto: 自动选择
    if apple_silicon:
        try:
            import mlx_whisper  # noqa: F401
            print("🍎 自动选择: mlx-whisper（Apple GPU/ANE 加速）")
            return MLXWhisperBackend()
        except ImportError:
            print("⚠️  mlx-whisper 未安装，回退到 faster-whisper")
            print("    💡 安装 mlx-whisper 可获得 2-4 倍加速: pip install mlx-whisper")
            return FasterWhisperBackend()
    else:
        print("💻 自动选择: faster-whisper（非 Apple Silicon 设备）")
        return FasterWhisperBackend()


# ===================== 转写流程 =====================


def transcribe_single(
    backend: TranscriberBackend,
    audio_path: str,
    output_path: str,
    model_size: str,
    language: str,
    time_offset: float = 0.0,
    **backend_kwargs,
) -> None:
    """转写单个音频文件。"""
    backend.transcribe(
        audio_path=audio_path,
        output_path=output_path,
        model_size=model_size,
        language=language,
        time_offset=time_offset,
        **backend_kwargs,
    )


def transcribe_with_split(
    backend: TranscriberBackend,
    audio_path: str,
    output_path: str,
    model_size: str,
    language: str,
    chunks: List[Tuple[str, float]],
    keep_chunks: bool = False,
    **backend_kwargs,
) -> None:
    """逐个转写拆分后的 chunk，最后合并。"""
    total_chunks = len(chunks)
    chunk_outputs = []

    print(f"\n🎤 开始逐段转写（共 {total_chunks} 段）\n{'=' * 50}")

    overall_start = time.time()

    for i, (chunk_path, offset) in enumerate(chunks):
        chunk_out = chunk_path.replace(".wav", ".txt")
        prefix = f"[{i + 1}/{total_chunks}]"

        print(f"\n{prefix} 转写第 {i + 1} 段 "
              f"（偏移: {_fmt_time(offset)}，{os.path.basename(chunk_path)}）")

        try:
            transcribe_single(
                backend=backend,
                audio_path=chunk_path,
                output_path=chunk_out,
                model_size=model_size,
                language=language,
                time_offset=offset,
                **backend_kwargs,
            )
            chunk_outputs.append(chunk_out)
        except Exception as e:
            print(f"   ❌ 转写失败: {e}", file=sys.stderr)

        elapsed = time.time() - overall_start
        done = i + 1
        eta = (elapsed / done) * (total_chunks - done)
        print(f"{prefix} 进度: {done}/{total_chunks} | "
              f"已耗时 {int(elapsed // 60)}:{int(elapsed % 60):02d} | "
              f"预计剩余 {int(eta // 60)}:{int(eta % 60):02d}")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 50}")
    print(f"✅ 全部转写完成！总耗时 {int(overall_elapsed // 60)} 分 {int(overall_elapsed % 60)} 秒")

    # 合并
    merge_transcripts(chunk_outputs, output_path)

    # 清理 chunk
    if not keep_chunks:
        print("🧹 清理临时文件...")
        for chunk_path, _ in chunks:
            if os.path.isfile(chunk_path):
                os.remove(chunk_path)
        for out in chunk_outputs:
            if os.path.isfile(out):
                os.remove(out)


# ===================== 主流程 =====================


def main():
    parser = argparse.ArgumentParser(
        description="视频/音频 → 文字转写插件（支持 Apple GPU 加速）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动选择最优后端（Apple Silicon → mlx GPU 加速）
  python transcribe.py video.mp4

  # 直接转写音频
  python transcribe.py audio.wav

  # 指定模型和语言
  python transcribe.py video.mp4 -m large-v3 -l en

  # 手动指定后端
  python transcribe.py video.mp4 --backend mlx    # 强制 MLX
  python transcribe.py video.mp4 --backend faster # 强制 faster-whisper

  # 拆分转写（超长音频推荐）
  python transcribe.py long_audio.wav --split 4           # 均分为 4 段
  python transcribe.py long_audio.wav --split-duration 30 # 每 30 分钟一段

  # 代理设置
  python transcribe.py video.mp4 --proxy http://127.0.0.1:7890  # 自定义代理
  python transcribe.py video.mp4 --no-proxy                      # 直连 HuggingFace

速度参考（Apple Silicon M 系列）:
  mlx-whisper tiny     ~ 50-80x 实时  （2小时 → ~2分钟）
  mlx-whisper medium   ~ 15-25x 实时  （2小时 → ~6分钟）
  mlx-whisper large-v3 ~  5-10x 实时  （2小时 → ~15分钟）
  faster-whisper       ~  1-2x  实时  （CPU int8）
        """,
    )

    # --- 输入/输出 ---
    parser.add_argument("input", help="输入文件路径（视频或音频）")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文本文件路径（默认: 与输入同目录同名 .txt）")

    # --- 模型 ---
    parser.add_argument("-m", "--model", choices=MODEL_CHOICES, default="medium",
                        help="Whisper 模型大小（默认: medium）")
    parser.add_argument("-l", "--language", default="zh",
                        help="音频语言代码（默认: zh；自动检测用 auto）")

    # --- 后端 ---
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto",
                        help="转写后端（默认: auto；Apple Silicon → mlx，其他 → faster）")

    # --- 拆分 ---
    split = parser.add_argument_group("音频拆分（适合超长录音）")
    split.add_argument("--split", type=int, default=0, metavar="N",
                       help="将音频平均拆分为 N 份，分别转写后合并")
    split.add_argument("--split-duration", type=float, default=0, metavar="MIN",
                       help="按指定时长（分钟）拆分音频后分别转写")
    split.add_argument("--keep-chunks", action="store_true",
                       help="保留拆分后的中间文件（默认清理）")

    # --- 网络代理 ---
    net = parser.add_argument_group("网络代理（下载模型时使用）")
    net.add_argument("--proxy", default=None,
                     help="HTTP 代理地址（默认: 自动检测 Clash Verge → 127.0.0.1:7897）")
    net.add_argument("--no-proxy", action="store_true",
                     help="禁用代理，直连 HuggingFace")

    # --- faster-whisper 专属 ---
    fw = parser.add_argument_group("faster-whisper 选项")
    fw.add_argument("--device", default="cpu", choices=DEVICE_CHOICES,
                    help="推理设备（默认: cpu）")
    fw.add_argument("--compute-type", default="int8",
                    choices=["int8", "int8_float16", "float16", "float32"],
                    help="计算精度（默认: int8）")
    fw.add_argument("--beam-size", type=int, default=5,
                    help="Beam search 宽度（默认: 5）")
    fw.add_argument("--no-vad", action="store_true",
                    help="禁用 VAD 静音过滤")

    # --- 音频 ---
    audio = parser.add_argument_group("音频提取选项")
    audio.add_argument("--sample-rate", type=int, default=16000,
                       help="提取音频的采样率（默认: 16000）")
    audio.add_argument("--keep-wav", action="store_true",
                       help="保留中间 WAV 文件")
    audio.add_argument("--wav-path", default=None,
                       help="中间 WAV 文件路径（默认: 系统临时目录）")

    args = parser.parse_args()

    # --- 校验 ---
    if args.split > 0 and args.split_duration > 0:
        print("❌ --split 和 --split-duration 不能同时使用", file=sys.stderr)
        sys.exit(1)
    if args.split > 1 and args.split > 99:
        print("❌ --split 最大 99", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"❌ 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    # --- 设置代理（在加载任何模型之前） ---
    setup_proxy(proxy=args.proxy, no_proxy=args.no_proxy)

    # --- 确定输出路径 ---
    output_path = args.output or os.path.splitext(args.input)[0] + ".txt"

    # --- 确定音频路径 ---
    wav_path = None
    auto_cleanup_wav = False

    if is_video(args.input):
        wav_path = args.wav_path or os.path.join(
            tempfile.gettempdir(),
            os.path.splitext(os.path.basename(args.input))[0] + ".wav",
        )
        extract_audio(args.input, wav_path, sample_rate=args.sample_rate)
        audio_path = wav_path
        auto_cleanup_wav = not args.keep_wav and not args.wav_path
    elif is_audio(args.input):
        audio_path = args.input
    else:
        print(f"⚠️  未知格式: {args.input}，直接尝试转写…")
        audio_path = args.input

    # --- 创建后端 ---
    backend = detect_backend(args.backend)

    # --- 构建后端参数 ---
    backend_kwargs = {}
    if isinstance(backend, FasterWhisperBackend):
        backend_kwargs.update(
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
        )

    # --- 拆分 or 直接转写 ---
    split_chunks_dir = None

    try:
        if args.split > 1:
            # 平均拆分
            split_chunks_dir = os.path.join(
                tempfile.gettempdir(),
                f"transcribe_chunks_{os.getpid()}",
            )
            os.makedirs(split_chunks_dir, exist_ok=True)
            chunks = split_audio_evenly(audio_path, args.split, split_chunks_dir)

            transcribe_with_split(
                backend=backend,
                audio_path=audio_path,
                output_path=output_path,
                model_size=args.model,
                language=args.language,
                chunks=chunks,
                keep_chunks=args.keep_chunks,
                **backend_kwargs,
            )

        elif args.split_duration > 0:
            # 按时长拆分
            split_chunks_dir = os.path.join(
                tempfile.gettempdir(),
                f"transcribe_chunks_{os.getpid()}",
            )
            os.makedirs(split_chunks_dir, exist_ok=True)
            chunks = split_audio_by_duration(audio_path, args.split_duration, split_chunks_dir)

            transcribe_with_split(
                backend=backend,
                audio_path=audio_path,
                output_path=output_path,
                model_size=args.model,
                language=args.language,
                chunks=chunks,
                keep_chunks=args.keep_chunks,
                **backend_kwargs,
            )

        else:
            # 直接转写
            transcribe_single(
                backend=backend,
                audio_path=audio_path,
                output_path=output_path,
                model_size=args.model,
                language=args.language,
                **backend_kwargs,
            )

    finally:
        # 清理 WAV
        if auto_cleanup_wav and wav_path and os.path.isfile(wav_path):
            os.remove(wav_path)
            print(f"🧹 已删除临时文件: {wav_path}")
        # 清理 chunks 目录
        if split_chunks_dir and os.path.isdir(split_chunks_dir):
            try:
                os.rmdir(split_chunks_dir)  # 只删空目录
            except OSError:
                pass


if __name__ == "__main__":
    main()
