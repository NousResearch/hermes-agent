---
title: "AudioCraft 音频生成 — AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效"
sidebar_label: "AudioCraft 音频生成"
description: "AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效"
---
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
AudioCraft: MusicGen text-to-music, AudioGen text-to-sound.
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## 技能元数据
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
| | |
|---|---|
| | |
| 路径 | `skills/mlops/models/audiocraft` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `audiocraft`, `torch>=2.0.0`, `transformers>=4.30.0` |
| 标签 | `多模态、音频生成、文本生音乐、文本生音频、MusicGen` |
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## 参考：完整 SKILL.md
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. 这是代理在技能激活时看到的指令。
:::
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
Comprehensive guide to using Meta's AudioCraft for text-to-music and text-to-audio generation with MusicGen, AudioGen, and EnCodec.
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## When to use AudioCraft
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
**Use AudioCraft when:**
- Need to generate music from text descriptions
- Creating sound effects and environmental audio
- Building music generation applications
- Need melody-conditioned music generation
- Want stereo audio output
- Require controllable music generation with style transfer
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
**Key features:**
- **MusicGen**: Text-to-music generation with melody conditioning
- **AudioGen**: Text-to-sound effects generation
- **EnCodec**: High-fidelity neural audio codec
- **Multiple model sizes**: Small (300M) to Large (3.3B)
- **Stereo support**: Full stereo audio generation
- **Style conditioning**: MusicGen-Style for reference-based generation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
**Use alternatives instead:**
- **Stable Audio**: For longer commercial music generation
- **Bark**: For text-to-speech with music/sound effects
- **Riffusion**: For spectogram-based music generation
- **OpenAI Jukebox**: For raw audio generation with lyrics
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## Quick start
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Installation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```bash
# AudioCraft 音频生成
pip install audiocraft
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
pip install git+https://github.com/facebookresearch/audiocraft.git
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
pip install transformers torch torchaudio
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Basic text-to-music (AudioCraft)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
import torchaudio
from audiocraft.models import MusicGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model = MusicGen.get_pretrained('facebook/musicgen-small')
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model.set_generation_params(
    duration=8,  # seconds
    top_k=250,
    temperature=1.0
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
descriptions = ["happy upbeat electronic dance music with synths"]
wav = model.generate(descriptions)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
torchaudio.save("output.wav", wav[0].cpu(), sample_rate=32000)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Using HuggingFace Transformers
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.to("cuda")
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
inputs = processor(
    text=["80s pop track with bassy drums and synth"],
    padding=True,
    return_tensors="pt"
).to("cuda")
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
audio_values = model.generate(
    **inputs,
    do_sample=True,
    guidance_scale=3,
    max_new_tokens=256
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Text-to-sound with AudioGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from audiocraft.models import AudioGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model = AudioGen.get_pretrained('facebook/audiogen-medium')
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
model.set_generation_params(duration=5)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
descriptions = ["dog barking in a park with birds chirping"]
wav = model.generate(descriptions)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
torchaudio.save("sound.wav", wav[0].cpu(), sample_rate=16000)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## Core concepts
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Architecture overview
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
<!-- ascii-guard-ignore -->
```
AudioCraft Architecture:
┌──────────────────────────────────────────────────────────────┐
│                    Text Encoder (T5)                          │
│                         │                                     │
│                    Text Embeddings                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              Transformer Decoder (LM)                         │
│     Auto-regressively generates audio tokens                  │
│     Using efficient token interleaving patterns               │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                EnCodec Audio Decoder                          │
│        Converts tokens back to audio waveform                 │
└──────────────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Model variants
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| `musicgen-small` | 300M | Text-to-music | Quick generation |
| `musicgen-medium` | 1.5B | Text-to-music | Balanced |
| `musicgen-large` | 3.3B | Text-to-music | Best quality |
| `musicgen-melody` | 1.5B | Text + melody | Melody conditioning |
| `musicgen-melody-large` | 3.3B | Text + melody | Best melody |
| `musicgen-stereo-*` | Varies | Stereo output | Stereo generation |
| `musicgen-style` | 1.5B | Style transfer | Reference-based |
| `audiogen-medium` | 1.5B | Text-to-sound | Sound effects |
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Generation parameters
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | 8.0 | Length in seconds (1-120) |
| `top_k` | 250 | Top-k sampling |
| `top_p` | 0.0 | Nucleus sampling (0 = disabled) |
| `temperature` | 1.0 | Sampling temperature |
| `cfg_coef` | 3.0 | Classifier-free guidance |
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## MusicGen usage
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Text-to-music generation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from audiocraft.models import MusicGen
import torchaudio
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
model = MusicGen.get_pretrained('facebook/musicgen-medium')
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model.set_generation_params(
    duration=30,          # Up to 30 seconds
    top_k=250,            # Sampling diversity
    top_p=0.0,            # 0 = use top_k only
    temperature=1.0,      # Creativity (higher = more varied)
    cfg_coef=3.0          # Text adherence (higher = stricter)
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
descriptions = [
    "epic orchestral soundtrack with strings and brass",
    "chill lo-fi hip hop beat with jazzy piano",
    "energetic rock song with electric guitar"
]
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
wav = model.generate(descriptions)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
for i, audio in enumerate(wav):
    torchaudio.save(f"music_{i}.wav", audio.cpu(), sample_rate=32000)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Melody-conditioned generation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from audiocraft.models import MusicGen
import torchaudio
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=30)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
melody, sr = torchaudio.load("melody.wav")
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
descriptions = ["acoustic guitar folk song"]
wav = model.generate_with_chroma(descriptions, melody, sr)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
torchaudio.save("melody_conditioned.wav", wav[0].cpu(), sample_rate=32000)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Stereo generation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from audiocraft.models import MusicGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model = MusicGen.get_pretrained('facebook/musicgen-stereo-medium')
model.set_generation_params(duration=15)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
descriptions = ["ambient electronic music with wide stereo panning"]
wav = model.generate(descriptions)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
print(f"Stereo shape: {wav.shape}")  # [1, 2, 480000]
torchaudio.save("stereo.wav", wav[0].cpu(), sample_rate=32000)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Audio continuation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
import torchaudio
audio, sr = torchaudio.load("intro.wav")
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
inputs = processor(
    audio=audio.squeeze().numpy(),
    sampling_rate=sr,
    text=["continue with a epic chorus"],
    padding=True,
    return_tensors="pt"
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=512)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## MusicGen-Style usage
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Style-conditioned generation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from audiocraft.models import MusicGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model = MusicGen.get_pretrained('facebook/musicgen-style')
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model.set_generation_params(
    duration=30,
    cfg_coef=3.0,
    cfg_coef_beta=5.0  # Style influence
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model.set_style_conditioner_params(
    eval_q=3,          # RVQ quantizers (1-6)
    excerpt_length=3.0  # Style excerpt length
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
style_audio, sr = torchaudio.load("reference_style.wav")
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
descriptions = ["upbeat dance track"]
wav = model.generate_with_style(descriptions, style_audio, sr)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Style-only generation (no text)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
# AudioCraft 音频生成
model.set_generation_params(
    duration=30,
    cfg_coef=3.0,
    cfg_coef_beta=None  # Disable double CFG for style-only
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
wav = model.generate_with_style([None], style_audio, sr)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## AudioGen usage
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Sound effect generation
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from audiocraft.models import AudioGen
import torchaudio
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=10)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
descriptions = [
    "thunderstorm with heavy rain and lightning",
    "busy city traffic with car horns",
    "ocean waves crashing on rocks",
    "crackling campfire in forest"
]
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
wav = model.generate(descriptions)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
for i, audio in enumerate(wav):
    torchaudio.save(f"sound_{i}.wav", audio.cpu(), sample_rate=16000)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## EnCodec usage
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Audio compression
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
from audiocraft.models import CompressionModel
import torch
import torchaudio
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model = CompressionModel.get_pretrained('facebook/encodec_32khz')
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
wav, sr = torchaudio.load("audio.wav")
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
if sr != 32000:
    resampler = torchaudio.transforms.Resample(sr, 32000)
    wav = resampler(wav)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
with torch.no_grad():
    encoded = model.encode(wav.unsqueeze(0))
    codes = encoded[0]  # Audio codes
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
with torch.no_grad():
    decoded = model.decode(codes)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
torchaudio.save("reconstructed.wav", decoded[0].cpu(), sample_rate=32000)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## Common workflows
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Workflow 1: Music generation pipeline
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
import torch
import torchaudio
from audiocraft.models import MusicGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
class MusicGenerator:
    def __init__(self, model_name="facebook/musicgen-medium"):
        self.model = MusicGen.get_pretrained(model_name)
        self.sample_rate = 32000
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    def generate(self, prompt, duration=30, temperature=1.0, cfg=3.0):
        self.model.set_generation_params(
            duration=duration,
            top_k=250,
            temperature=temperature,
            cfg_coef=cfg
        )
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
        with torch.no_grad():
            wav = self.model.generate([prompt])
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
        return wav[0].cpu()
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    def generate_batch(self, prompts, duration=30):
        self.model.set_generation_params(duration=duration)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
        with torch.no_grad():
            wav = self.model.generate(prompts)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
        return wav.cpu()
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    def save(self, audio, path):
        torchaudio.save(path, audio, sample_rate=self.sample_rate)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
generator = MusicGenerator()
audio = generator.generate(
    "epic cinematic orchestral music",
    duration=30,
    temperature=1.0
)
generator.save(audio, "epic_music.wav")
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Workflow 2: Sound design batch processing
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
import json
from pathlib import Path
from audiocraft.models import AudioGen
import torchaudio
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
def batch_generate_sounds(sound_specs, output_dir):
    """
    Generate multiple sounds from specifications.
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    Args:
        sound_specs: list of {"name": str, "description": str, "duration": float}
        output_dir: output directory path
    """
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    results = []
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    for spec in sound_specs:
        model.set_generation_params(duration=spec.get("duration", 5))
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
        wav = model.generate([spec["description"]])
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
        output_path = output_dir / f"{spec['name']}.wav"
        torchaudio.save(str(output_path), wav[0].cpu(), sample_rate=16000)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
        results.append({
            "name": spec["name"],
            "path": str(output_path),
            "description": spec["description"]
        })
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    return results
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
sounds = [
    {"name": "explosion", "description": "massive explosion with debris", "duration": 3},
    {"name": "footsteps", "description": "footsteps on wooden floor", "duration": 5},
    {"name": "door", "description": "wooden door creaking and closing", "duration": 2}
]
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
results = batch_generate_sounds(sounds, "sound_effects/")
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Workflow 3: Gradio demo
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
import gradio as gr
import torch
import torchaudio
from audiocraft.models import MusicGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
model = MusicGen.get_pretrained('facebook/musicgen-small')
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
def generate_music(prompt, duration, temperature, cfg_coef):
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        cfg_coef=cfg_coef
    )
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    with torch.no_grad():
        wav = model.generate([prompt])
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
    # Save to temp file
    path = "temp_output.wav"
    torchaudio.save(path, wav[0].cpu(), sample_rate=32000)
    return path
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Music Description", placeholder="upbeat electronic dance music"),
        gr.Slider(1, 30, value=8, label="Duration (seconds)"),
        gr.Slider(0.5, 2.0, value=1.0, label="Temperature"),
        gr.Slider(1.0, 10.0, value=3.0, label="CFG Coefficient")
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="MusicGen Demo"
)
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
demo.launch()
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## Performance optimization
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Memory optimization
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
# AudioCraft 音频生成
model = MusicGen.get_pretrained('facebook/musicgen-small')
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
torch.cuda.empty_cache()
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model.set_generation_params(duration=10)  # Instead of 30
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
model = model.half()
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### Batch processing efficiency
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
```python
# AudioCraft 音频生成
descriptions = ["prompt1", "prompt2", "prompt3", "prompt4"]
wav = model.generate(descriptions)  # Single batch
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
# AudioCraft 音频生成
for desc in descriptions:
    wav = model.generate([desc])  # Multiple batches (slower)
```
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
### GPU memory requirements
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
| Model | FP32 VRAM | FP16 VRAM |
|-------|-----------|-----------|
| musicgen-small | ~4GB | ~2GB |
| musicgen-medium | ~8GB | ~4GB |
| musicgen-large | ~16GB | ~8GB |
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## Common issues
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
| Issue | Solution |
|-------|----------|
| CUDA OOM | Use smaller model, reduce duration |
| Poor quality | Increase cfg_coef, better prompts |
| Generation too short | Check max duration setting |
| Audio artifacts | Try different temperature |
| Stereo not working | Use stereo model variant |
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## References
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
- **[Advanced Usage](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/audiocraft/references/advanced-usage.md)** - Training, fine-tuning, deployment
- **[Troubleshooting](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/audiocraft/references/troubleshooting.md)** - Common issues and solutions
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
## Resources
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。
- **GitHub**: https://github.com/facebookresearch/audiocraft
- **Paper (MusicGen)**: https://arxiv.org/abs/2306.05284
- **Paper (AudioGen)**: https://arxiv.org/abs/2209.15352
- **HuggingFace**: https://huggingface.co/facebook/musicgen-small
- **Demo**: https://huggingface.co/spaces/facebook/MusicGen
AudioCraft：MusicGen 文本生音乐、AudioGen 文本生音效。