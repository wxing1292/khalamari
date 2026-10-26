# Embedded Silero VAD

`silero_vad_v6.onnx` is the fixed Silero v6 model used by the ASR client.
Rust embeds its bytes in the executable. ONNX Runtime loads the graph directly from memory.

| Property | Value |
| --- | --- |
| Source | [SYSTRAN/faster-whisper v1.2.1](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/assets/silero_vad_v6.onnx) |
| Size | 1,245,151 bytes |
| SHA-256 | `4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2` |
| Licenses | [Silero MIT notice](LICENSE.silero), [faster-whisper MIT notice](LICENSE.faster-whisper) |

Model updates must preserve or update the initialization contract in [`src/models.rs`](../src/models.rs).
Run the native model reference checks and recording acceptance before changing the model.
See [Qwen3-ASR](../../../docs/qwen3_asr.md#native-model-setup) for build and runtime ownership.
