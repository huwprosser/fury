import wave

import numpy as np

from fury import Agent


def write_wav(path: str, audio: np.ndarray, sample_rate: int = 24000) -> None:
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def test_tts_generation(tmp_path):
    agent = Agent(
        model="unsloth/GLM-4.6V-Flash-GGUF:Q8_0",
        system_prompt="You are a helpful assistant.",
    )

    audio_chunks = list(
        agent.speak(
            text="Hello from Fury!",
            ref_text="Welcome home sir.",
            ref_audio_path="examples/resources/ref.wav",
        )
    )

    assert audio_chunks

    audio = np.concatenate(audio_chunks)
    output_path = tmp_path / "output.wav"
    write_wav(str(output_path), audio)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
