#!/usr/bin/env python3
"""
视频/音频 → 文字转写

后端:
  mlx-whisper   — Apple Silicon GPU / ANE 加速（M 系列 Mac 首选）
  faster-whisper — CTranslate2 CPU 推理
  auto           — 自动检测（Apple Silicon → mlx，其他 → faster）

模型: tiny / small / medium / large-v3

用法:
  python transcribe.py video.mp4                     # 基础转写
  python transcribe.py audio.wav -m large-v3 -l en   # 指定模型和语言
  python transcribe.py long.wav --split 4            # 拆分转写
  python transcribe.py video.mp4 --proxy http://127.0.0.1:7897  # 走代理下载模型
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

# ===================== 常量 =====================

MODEL_CHOICES      = ["tiny", "small", "medium", "large-v3"]
BACKEND_CHOICES    = ["auto", "mlx", "faster"]
DEVICE_CHOICES     = ["cpu", "cuda"]

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".wmv", ".m4v", ".ts"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


# ===================== 1. 文件类型检测 =====================

def is_video(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in VIDEO_EXTS


def is_audio(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in AUDIO_EXTS


# ===================== 2. 音频提取（视频 → WAV） =====================

def extract_audio(video_path: str, wav_path: str, sample_rate: int = 16000) -> None:
    """用 ffmpeg 从视频中提取单声道 16kHz WAV。"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-sample_fmt", "s16", "-y", wav_path,
    ]
    print(f"🎬 提取音频: {video_path}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg 提取失败:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ 音频已保存: {wav_path}")


# ===================== 3. 音频信息（时长） =====================

def get_audio_duration(audio_path: str) -> float:
    """获取音频总时长（秒）。"""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", audio_path]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return 0.0

    duration = float(info.get("format", {}).get("duration", 0))
    if duration <= 0:
        for stream in info.get("streams", []):
            d = float(stream.get("duration", 0))
            if d > 0:
                duration = d
                break
    return duration


# ===================== 4. 音频拆分（可选功能） =====================

def _fmt_time(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def split_audio_evenly(audio_path: str, num_parts: int,
                       output_dir: str) -> List[Tuple[str, float]]:
    """均分为 N 份，返回 [(chunk_path, offset_seconds), ...]."""
    total = get_audio_duration(audio_path)
    chunk_dur = total / num_parts

    print(f"\n✂️  音频总时长: {total / 60:.1f} 分钟")
    print(f"   均分为 {num_parts} 份，每份约 {chunk_dur / 60:.1f} 分钟\n")

    chunks = []
    for i in range(num_parts):
        start = i * chunk_dur
        chunk_path = os.path.join(output_dir, f"chunk_{i + 1:03d}.wav")
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ss", str(start), "-t", str(chunk_dur),
            "-c", "copy", "-y", chunk_path,
        ], check=True, capture_output=True, text=True)

        actual = get_audio_duration(chunk_path)
        chunks.append((chunk_path, start))
        print(f"   Chunk {i + 1}/{num_parts}: "
              f"[{_fmt_time(start)} -> {_fmt_time(start + actual)}] "
              f"→ {os.path.basename(chunk_path)}")
    return chunks


def split_audio_by_duration(audio_path: str, duration_min: float,
                            output_dir: str) -> List[Tuple[str, float]]:
    """按固定时长（分钟）拆分。"""
    total = get_audio_duration(audio_path)
    chunk_sec = duration_min * 60
    num_parts = max(1, math.ceil(total / chunk_sec))

    print(f"\n✂️  音频总时长: {total / 60:.1f} 分钟")
    print(f"   按每段 {duration_min} 分钟拆分，共 {num_parts} 段\n")

    chunks = []
    for i in range(num_parts):
        start = i * chunk_sec
        chunk_path = os.path.join(output_dir, f"chunk_{i + 1:03d}.wav")
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ss", str(start), "-t", str(chunk_sec),
            "-c", "copy", "-y", chunk_path,
        ], check=True, capture_output=True, text=True)

        actual = get_audio_duration(chunk_path)
        chunks.append((chunk_path, start))
        print(f"   Chunk {i + 1}/{num_parts}: "
              f"[{_fmt_time(start)} -> {_fmt_time(start + actual)}] "
              f"→ {os.path.basename(chunk_path)}")
    return chunks


# ===================== 5. 网络代理（可选功能） =====================

def setup_proxy(proxy: Optional[str]) -> None:
    """设置 HTTP/HTTPS 代理（仅当用户显式指定 --proxy 时生效）。

    不传 --proxy 则不设任何代理，直连访问 HuggingFace。
    """
    if not proxy:
        return

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = proxy
    print(f"🔗 使用代理: {proxy}")


# ===================== 6. 设备检测 =====================

def is_apple_silicon() -> bool:
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


# ===================== 7. 转写后端 =====================

class TranscriberBackend(ABC):
    name: str = ""

    @abstractmethod
    def transcribe(self, audio_path: str, output_path: str,
                   model_size: str, language: str, time_offset: float, **kwargs) -> None:
        ...


class FasterWhisperBackend(TranscriberBackend):
    name = "faster-whisper"

    def transcribe(self, audio_path, output_path, model_size, language,
                   time_offset=0.0, device="cpu", compute_type="int8",
                   beam_size=5, vad_filter=True, **kwargs):
        from faster_whisper import WhisperModel

        print(f"📦 加载模型: {model_size}（{self.name} | {device}/{compute_type}）...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        print(f"🎤 开始转写...")
        start = time.time()

        segments, info = model.transcribe(
            audio_path, language=language,
            beam_size=beam_size, vad_filter=vad_filter,
        )

        print(f"🌐 检测语言: {info.language}（概率: {info.language_probability:.2%}）\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                abs_s = seg.start + time_offset
                abs_e = seg.end + time_offset
                m1, s1 = divmod(abs_s, 60)
                m2, s2 = divmod(abs_e, 60)
                line = f"[{int(m1):02d}:{s1:05.2f} -> {int(m2):02d}:{s2:05.2f}] {seg.text.strip()}"
                f.write(line + "\n")
                print(line)

        elapsed = time.time() - start
        print(f"\n✅ 完成！耗时 {int(elapsed // 60)} 分 {int(elapsed % 60)} 秒")


class MLXWhisperBackend(TranscriberBackend):
    name = "mlx-whisper"

    MODEL_MAP = {
        "tiny":     "mlx-community/whisper-tiny",
        "small":    "mlx-community/whisper-small-mlx",
        "medium":   "mlx-community/whisper-medium",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    }

    def transcribe(self, audio_path, output_path, model_size, language,
                   time_offset=0.0, **kwargs):
        import mlx_whisper

        model_path = self.MODEL_MAP.get(model_size, model_size)
        print(f"📦 加载模型: {model_size}（{self.name} | Apple GPU/ANE | {model_path}）...")

        lang_arg = language if language and language != "auto" else None
        print(f"🎤 开始转写（语言: {lang_arg or '自动检测'}）...")
        start = time.time()

        result = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=model_path,
            language=lang_arg, verbose=True,
            **kwargs,
        )

        detected = result.get("language", "?")
        print(f"\n🌐 检测语言: {detected}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            for seg in result.get("segments", []):
                abs_s = seg.get("start", 0) + time_offset
                abs_e = seg.get("end", 0) + time_offset
                text = seg.get("text", "").strip()
                m1, s1 = divmod(abs_s, 60)
                m2, s2 = divmod(abs_e, 60)
                line = f"[{int(m1):02d}:{s1:05.2f} -> {int(m2):02d}:{s2:05.2f}] {text}"
                f.write(line + "\n")
                print(line)

        elapsed = time.time() - start
        print(f"\n✅ 完成！耗时 {int(elapsed // 60)} 分 {int(elapsed % 60)} 秒")


def create_backend(choice: str) -> TranscriberBackend:
    """后端工厂。auto 模式自动选择最优。"""
    apple = is_apple_silicon()

    if choice == "mlx":
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            print("❌ mlx-whisper 未安装: pip install mlx-whisper", file=sys.stderr)
            sys.exit(1)
        print("🍎 使用 mlx-whisper（Apple GPU/ANE）")
        return MLXWhisperBackend()

    if choice == "faster":
        print("💻 使用 faster-whisper（CTranslate2）")
        return FasterWhisperBackend()

    # auto
    if apple:
        try:
            import mlx_whisper  # noqa: F401
            print("🍎 自动选择: mlx-whisper（Apple GPU/ANE）")
            return MLXWhisperBackend()
        except ImportError:
            print("⚠️  mlx-whisper 未安装，回退 faster-whisper")
            return FasterWhisperBackend()
    else:
        print("💻 自动选择: faster-whisper")
        return FasterWhisperBackend()


# ===================== 8. 结果合并 =====================

def merge_transcripts(chunk_outputs: List[str], final_output: str) -> None:
    """将多个 split chunk 的转写结果合并为一个文件。"""
    print(f"\n📦 合并 {len(chunk_outputs)} 个转写结果...")
    with open(final_output, "w", encoding="utf-8") as out:
        for i, path in enumerate(chunk_outputs):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out.write(f.read())
                if i < len(chunk_outputs) - 1:
                    out.write("\n")
                print(f"   ✅ Chunk {i + 1}/{len(chunk_outputs)}")
            except FileNotFoundError:
                print(f"   ⚠️  Chunk {i + 1} 结果丢失", file=sys.stderr)
    print(f"📄 最终结果: {final_output}")


# ===================== 9. 清理 =====================

def cleanup_temp_files(*paths: str) -> None:
    """删除临时文件/目录（忽略不存在的）。"""
    for p in paths:
        if p and os.path.isfile(p):
            os.remove(p)
        elif p and os.path.isdir(p):
            try:
                os.rmdir(p)
            except OSError:
                pass


# ===================== 10. 单段转写（无拆分） =====================

def transcribe_single(backend: TranscriberBackend, audio_path: str,
                      output_path: str, model_size: str, language: str,
                      **backend_kwargs):
    backend.transcribe(
        audio_path=audio_path, output_path=output_path,
        model_size=model_size, language=language, time_offset=0.0,
        **backend_kwargs,
    )
    print(f"📄 结果已保存: {output_path}")


# ===================== 11. 拆分转写 =====================

def transcribe_with_split(backend: TranscriberBackend, chunks, output_path: str,
                          model_size: str, language: str,
                          keep_chunks: bool = False, **backend_kwargs):
    """逐个转写 chunk，合并，清理。"""
    total = len(chunks)
    overall_start = time.time()
    chunk_outputs = []

    print(f"\n🎤 开始逐段转写（共 {total} 段）\n{'=' * 50}")

    for i, (chunk_path, offset) in enumerate(chunks):
        chunk_out = chunk_path.replace(".wav", ".txt")
        prefix = f"[{i + 1}/{total}]"

        print(f"\n{prefix} 转写第 {i + 1} 段（偏移: {_fmt_time(offset)}）")
        try:
            backend.transcribe(
                audio_path=chunk_path, output_path=chunk_out,
                model_size=model_size, language=language, time_offset=offset,
                **backend_kwargs,
            )
            chunk_outputs.append(chunk_out)
        except Exception as e:
            print(f"   ❌ 失败: {e}", file=sys.stderr)

        elapsed = time.time() - overall_start
        done = i + 1
        eta = (elapsed / done) * (total - done) if done < total else 0
        print(f"{prefix} 进度: {done}/{total} | "
              f"已耗时 {int(elapsed // 60)}:{int(elapsed % 60):02d} | "
              f"预计剩余 {int(eta // 60)}:{int(eta % 60):02d}")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 50}")
    print(f"✅ 全部转写完成！总耗时 {int(overall_elapsed // 60)} 分 {int(overall_elapsed % 60)} 秒")

    merge_transcripts(chunk_outputs, output_path)

    if not keep_chunks:
        print("🧹 清理临时文件...")
        for p, _ in chunks:
            cleanup_temp_files(p, p.replace(".wav", ".txt"))


# ===================== 12. 主入口 =====================

def main():
    parser = argparse.ArgumentParser(
        description="视频/音频 → 文字转写（基于 Whisper，Apple GPU 加速）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python transcribe.py video.mp4
  python transcribe.py audio.wav -m large-v3 -l en
  python transcribe.py long.wav --split 4
  python transcribe.py long.wav --split-duration 30
  python transcribe.py video.mp4 --proxy http://127.0.0.1:7897
        """,
    )
    parser.add_argument("input", help="输入文件（视频或音频）")
    parser.add_argument("-o", "--output", default=None, help="输出文件（默认: 输入文件名.txt）")
    parser.add_argument("-m", "--model", choices=MODEL_CHOICES, default="medium", help="模型大小")
    parser.add_argument("-l", "--language", default="zh", help="语言代码（zh/en/auto...）")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto", help="推理后端")

    # 拆分（可选）
    split = parser.add_argument_group("拆分（可选）")
    split.add_argument("--split", type=int, default=0, metavar="N", help="均分为 N 份")
    split.add_argument("--split-duration", type=float, default=0, metavar="MIN", help="按分钟拆分")
    split.add_argument("--keep-chunks", action="store_true", help="保留中间片段")

    # 代理（可选）
    net = parser.add_argument_group("网络代理（可选）")
    net.add_argument("--proxy", default=None, help="HTTP 代理地址，如 http://127.0.0.1:7897")

    # 音频提取（可选）
    audio = parser.add_argument_group("音频提取（可选）")
    audio.add_argument("--sample-rate", type=int, default=16000, help="采样率")
    audio.add_argument("--keep-wav", action="store_true", help="保留提取的 WAV")
    audio.add_argument("--wav-path", default=None, help="WAV 保存路径")

    # faster-whisper 专属
    fw = parser.add_argument_group("faster-whisper 选项")
    fw.add_argument("--device", default="cpu", choices=DEVICE_CHOICES, help="推理设备")
    fw.add_argument("--compute-type", default="int8",
                    choices=["int8", "int8_float16", "float16", "float32"], help="计算精度")
    fw.add_argument("--beam-size", type=int, default=5, help="Beam search 宽度")
    fw.add_argument("--no-vad", action="store_true", help="禁用 VAD 静音过滤")

    args = parser.parse_args()

    # --- 校验 ---
    if args.split > 0 and args.split_duration > 0:
        print("❌ --split 和 --split-duration 不能同时用", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"❌ 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    # --- 网络代理（仅当用户指定时启用） ---
    setup_proxy(args.proxy)

    output_path = args.output or os.path.splitext(args.input)[0] + ".txt"

    # --- 步骤 1: 准备音频 ---
    wav_path = None
    cleanup_wav = False

    if is_video(args.input):
        wav_path = args.wav_path or os.path.join(
            tempfile.gettempdir(),
            os.path.splitext(os.path.basename(args.input))[0] + ".wav",
        )
        extract_audio(args.input, wav_path, args.sample_rate)
        audio_path = wav_path
        cleanup_wav = not args.keep_wav and not args.wav_path
    else:
        audio_path = args.input

    # --- 步骤 2: 创建后端 ---
    backend = create_backend(args.backend)

    backend_kwargs = {}
    if isinstance(backend, FasterWhisperBackend):
        backend_kwargs.update(
            device=args.device, compute_type=args.compute_type,
            beam_size=args.beam_size, vad_filter=not args.no_vad,
        )

    # --- 步骤 3: 转写（拆分 or 直接） ---
    chunks_dir = None

    try:
        if args.split > 1:
            chunks_dir = os.path.join(tempfile.gettempdir(), f"transcribe_chunks_{os.getpid()}")
            os.makedirs(chunks_dir, exist_ok=True)
            chunks = split_audio_evenly(audio_path, args.split, chunks_dir)
            transcribe_with_split(backend, chunks, output_path,
                                  args.model, args.language,
                                  keep_chunks=args.keep_chunks, **backend_kwargs)

        elif args.split_duration > 0:
            chunks_dir = os.path.join(tempfile.gettempdir(), f"transcribe_chunks_{os.getpid()}")
            os.makedirs(chunks_dir, exist_ok=True)
            chunks = split_audio_by_duration(audio_path, args.split_duration, chunks_dir)
            transcribe_with_split(backend, chunks, output_path,
                                  args.model, args.language,
                                  keep_chunks=args.keep_chunks, **backend_kwargs)

        else:
            transcribe_single(backend, audio_path, output_path,
                            args.model, args.language, **backend_kwargs)

    finally:
        cleanup_temp_files(wav_path if cleanup_wav else None, chunks_dir)


if __name__ == "__main__":
    main()
