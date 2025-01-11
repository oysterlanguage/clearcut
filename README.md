# clearcut: Ultra-Precise Audio Segmentation
**clearcut** is an advanced audio processing tool designed for ultra-clean audio segmentation. Whether you're working on speech recognition, transcription alignment, or other audio-based applications, ClearCut ensures your segments are precise, clean, and never clip the first or last word of a segment.

Orchestrates the entire pipeline:  
  1. **Standardizes** audio (resampling, volume normalization)  
  2. **Optionally separates** vocals from instrumentals  
  3. **Performs ASR** (using WhisperX) or **loads user transcripts** if provided (`.txt` files)  
  4. **Detects breath intervals** using `respiro.py`  
  5. **Aligns** transcripts and breath intervals  
  6. **Segments** audio based on threshold crossing & minima detection  
  7. **Writes** out training text files and optional TextGrid annotations

## Installation

Clone this repository:
```bash
git clone https://github.com/your-username/audio-segmentation-alignment.git
cd audio-segmentation-alignment
```

download models:

https://github.com/ydqmkkx/Respiro-en/blob/main/respiro-en.pt to your models directory
https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx to your models directory


Create and activate a Python virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
# or on Windows: venv\Scripts\activate
```

**Install dependencies**:
```bash
pip install -r requirements.txt
```
Some of these libraries (e.g., onnxruntime-gpu, torch) might require specialized wheels if you plan to run on GPU. Please consult the official PyTorch and onnxruntime documentation for platform-specific instructions.


After installing, you can run the entire pipeline via:
```bash
python main.py --config config/config.yaml
```

## Acknowledgement 🔔
This project would no be possible without the work by these excellent developers!
- ASR (WhisperX): [WhisperX](https://github.com/m-bain/whisperX)
- Source Separation: [UVR-MDX-NET-Inst_HQ_3](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)
- Breath Detection: [respiro.py](https://github.com/ydqmkkx/Respiro-en)
- Emelia: https://github.com/open-mmlab/Amphion/blob/main/preprocessors/Emilia/README.md
- fairseq: https://github.com/facebookresearch/fairseq/blob/main/examples/mms/data_prep/README.md