"""
NeuTTS Minimal inference script.
"""

import os
import glob
import platform
import logging
import numpy as np
import torch
import re
import warnings
from typing import Generator, List, Optional

from .utils.audio import load_audio
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from neucodec import NeuCodec, NeuCodecOnnxDecoder
from llama_cpp import Llama
from llama_cpp._logger import logger as llama_logger

warnings.filterwarnings("ignore")


def _configure_espeak_library():
    """Auto-detect and configure espeak library on macOS."""
    if platform.system() != "Darwin":
        return  # Only needed on macOS

    # Common Homebrew installation paths
    search_paths = [
        "/opt/homebrew/Cellar/espeak/*/lib/libespeak.*.dylib",  # Apple Silicon
        "/usr/local/Cellar/espeak/*/lib/libespeak.*.dylib",  # Intel
        "/opt/homebrew/bin/espeak/*/lib/libespeak.*.dylib",  # Fallback
        "/opt/homebrew/lib/libespeak.*.dylib",  # Another common location
    ]

    # Explicit check for the path seen in original code if generic search fails
    specific_path = "/opt/homebrew/bin/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib"
    if os.path.exists(specific_path):
        EspeakWrapper.set_library(specific_path)
        return

    for pattern in search_paths:
        matches = glob.glob(pattern)
        if matches:
            try:
                EspeakWrapper.set_library(matches[0])
                return
            except Exception:
                pass


_configure_espeak_library()


def _linear_overlap_add(frames: List[np.ndarray], stride: int) -> np.ndarray:
    assert len(frames)
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]

    total_size = 0
    for i, frame in enumerate(frames):
        frame_end = stride * i + frame.shape[-1]
        total_size = max(total_size, frame_end)

    sum_weight = np.zeros(total_size, dtype=dtype)
    out = np.zeros(*shape, total_size, dtype=dtype)

    offset: int = 0
    for frame in frames:
        frame_length = frame.shape[-1]
        t = np.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = np.abs(0.5 - (t - 0.5))

        out[..., offset : offset + frame_length] += weight * frame
        sum_weight[offset : offset + frame_length] += weight
        offset += stride

    # Avoid division by zero
    mask = sum_weight > 0
    out[..., mask] /= sum_weight[mask]
    return out


class NeuTTSMinimal:
    _SPEECH_TOKEN_RE = re.compile(r"<\|speech_(\d+)\|>")

    def __init__(
        self,
        backbone_path: str,
        codec_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_batch: int = 2048,
        verbose: bool = False,
        backbone_filename: str = "*.gguf",
    ):
        # Consts
        self.sample_rate = 24_000
        self.max_context = n_ctx
        self.hop_length = 480
        self.streaming_overlap_frames = 1
        self.streaming_frames_per_chunk = 25
        self.streaming_lookforward = 5
        self.streaming_lookback = 50
        self.streaming_stride_samples = (
            self.streaming_frames_per_chunk * self.hop_length
        )
        self._ref_codes: Optional[torch.Tensor] = None
        self._ref_audio_path: Optional[str] = None

        # Load Phonemizer
        print("Loading phonemizer...")
        self.phonemizer = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        if verbose:
            llama_logger.setLevel(logging.DEBUG)
        else:
            llama_logger.setLevel(logging.CRITICAL + 1)

        # Load Backbone (Llama GGUF)
        print(f"Loading backbone from {backbone_path}...")

        # Check if backbone_path is a local file path
        if os.path.isfile(backbone_path):
            self.backbone = Llama(
                model_path=backbone_path,
                verbose=False,
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.max_context,
                n_batch=n_batch,
                mlock=True,
                flash_attn=True if n_gpu_layers == -1 else False,
            )
        else:
            # Fallback to Hugging Face repository
            self.backbone = Llama.from_pretrained(
                repo_id=backbone_path,
                filename=backbone_filename,
                verbose=False,
                n_gpu_layers=n_gpu_layers,
                n_ctx=self.max_context,
                n_batch=n_batch,
                mlock=True,
                flash_attn=True if n_gpu_layers == -1 else False,
            )

        # Load Codec (ONNX)
        print(f"Loading codec from {codec_path}...")
        # If it's a repo ID, use from_pretrained, else local path
        if os.path.isfile(codec_path):
            self.codec = NeuCodecOnnxDecoder(codec_path)
        else:
            self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_path)

    def _to_phones(self, text: str) -> str:
        phones = self.phonemizer.phonemize([text])
        phones = phones[0].split()
        phones = " ".join(phones)
        return phones

    def _decode(self, codes: str) -> np.ndarray:
        """Decode from string format (legacy). Prefer _decode_ids for performance."""
        speech_ids = [int(num) for num in self._SPEECH_TOKEN_RE.findall(codes)]
        return self._decode_ids(speech_ids)

    @staticmethod
    def encode_reference_audio(
        ref_audio_path: str,
        *,
        output_path: Optional[str] = None,
        sample_rate: int = 16000,
        codec_id: str = "neuphonic/neucodec",
        device: str = "cpu",
    ) -> torch.Tensor:
        """Encode a reference audio file into NeuCodec speech token IDs."""
        codec = NeuCodec.from_pretrained(codec_id)
        codec.eval().to(device)

        wav, _ = load_audio(ref_audio_path, sr=sample_rate, mono=True)
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            ref_codes = (
                codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
            )

        ref_codes = ref_codes.detach().cpu()

        if output_path:
            if not output_path.endswith(".pt"):
                raise ValueError("Output path should end with .pt to save the codes.")
            torch.save(ref_codes, output_path)

        return ref_codes

    def _decode_ids(self, speech_ids: List[int]) -> np.ndarray:
        """Decode directly from integer speech token IDs (no string/regex overhead)."""
        if len(speech_ids) > 0:
            codes_arr = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]
            recon = self.codec.decode_code(codes_arr)
            return recon[0, 0, :]
        else:
            raise ValueError("No valid speech tokens to decode.")

    def infer_stream(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
    ) -> Generator[np.ndarray, None, None]:

        if self._ref_codes is None or self._ref_audio_path != ref_audio_path:
            self._ref_codes = self.encode_reference_audio(ref_audio_path)
            self._ref_audio_path = ref_audio_path

        # Preprocessing
        ref_phones = self._to_phones(ref_text)
        input_phones = self._to_phones(text)

        # Handle ref_codes format
        if isinstance(self._ref_codes, torch.Tensor):
            ref_codes_list = self._ref_codes.flatten().tolist()
        elif isinstance(self._ref_codes, np.ndarray):
            ref_codes_list = self._ref_codes.flatten().tolist()
        else:
            ref_codes_list = self._ref_codes

        codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes_list])

        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_phones} {input_phones}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
        )

        # Store integer IDs directly instead of formatted strings
        token_id_cache: List[int] = list(ref_codes_list)
        n_decoded_tokens: int = len(ref_codes_list)
        prev_frame: Optional[np.ndarray] = None

        # Streaming Generation
        for item in self.backbone(
            prompt,
            max_tokens=self.max_context,
            temperature=1.0,
            top_k=50,
            stop=["<|SPEECH_GENERATION_END|>"],
            stream=True,
        ):
            output_str = item["choices"][0]["text"]
            match = self._SPEECH_TOKEN_RE.search(output_str)
            if match:
                token_id_cache.append(int(match.group(1)))

            if (
                len(token_id_cache) - n_decoded_tokens
                >= self.streaming_frames_per_chunk + self.streaming_lookforward
            ):

                # decode chunk
                tokens_start = max(
                    n_decoded_tokens
                    - self.streaming_lookback
                    - self.streaming_overlap_frames,
                    0,
                )
                tokens_end = (
                    n_decoded_tokens
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )

                # Calculate sample ranges
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (
                        self.streaming_frames_per_chunk
                        + 2 * self.streaming_overlap_frames
                    )
                    * self.hop_length
                )

                curr_ids = token_id_cache[tokens_start:tokens_end]
                try:
                    recon = self._decode_ids(curr_ids)
                except ValueError:
                    continue

                recon = recon[sample_start:sample_end]

                # O(1) overlap-add: blend only with the previous frame
                if prev_frame is not None:
                    blended = _linear_overlap_add(
                        [prev_frame, recon], stride=self.streaming_stride_samples
                    )
                    processed_recon = blended[
                        self.streaming_stride_samples : 2
                        * self.streaming_stride_samples
                    ]
                else:
                    # First chunk: no previous frame to crossfade with
                    processed_recon = recon[: self.streaming_stride_samples]

                prev_frame = recon
                n_decoded_tokens += self.streaming_frames_per_chunk

                yield processed_recon

        # Final decoding
        remaining_tokens = len(token_id_cache) - n_decoded_tokens
        if remaining_tokens > 0:
            tokens_start = max(
                len(token_id_cache)
                - (
                    self.streaming_lookback
                    + self.streaming_overlap_frames
                    + remaining_tokens
                ),
                0,
            )
            sample_start = (
                len(token_id_cache)
                - tokens_start
                - remaining_tokens
                - self.streaming_overlap_frames
            ) * self.hop_length

            curr_ids = token_id_cache[tokens_start:]
            try:
                recon = self._decode_ids(curr_ids)
                recon = recon[sample_start:]

                if prev_frame is not None:
                    blended = _linear_overlap_add(
                        [prev_frame, recon], stride=self.streaming_stride_samples
                    )
                    processed_recon = blended[self.streaming_stride_samples :]
                else:
                    processed_recon = recon

                yield processed_recon
            except ValueError:
                pass
