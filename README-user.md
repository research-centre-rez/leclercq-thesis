# High-Resolution Registration of Irradiated Concrete Scans

This repository contains the codebase and supporting materials for the master’s thesis *“High-resolution Registration of Irradiated Concrete Scans”* by Erik Philippe Leclercq. It presents a complete image processing pipeline for registering high-resolution video scans of rotating concrete samples used in nuclear power plant research.

## 🎯 Purpose

The system enables:
- Remote analysis of radioactive concrete samples.
- Automatic video registration using state-of-the-art algorithms.
- Generation of composite images for visual inspection and automated crack detection.

## 🖼️ Input Data

- 4K video scans of cylindrical concrete samples (360° rotation).
- Two lighting conditions: **low-angle** (≤15°) and **front-lighting** (≥82°).
- Videos must contain one lighting condition per segment.

## 🧪 What It Does

1. **Splits videos** into light-specific segments.
2. **Performs optical flow-based rotation correction.**
3. **Applies registration algorithms** (µDIC, ORB, or LightGlue).
4. **Generates composite images** using min/max pixel fusion.
5. **Evaluates registration quality** with sharpness metrics.

## 📦 Installation

```bash
git clone https://github.com/research-centre-rez/leclercq-thesis.git
cd leclercq-thesis

# Downloads the LightGlue submodule
git submodule update --init
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupytext src/demo.py --to ipynb

Make sure FFmpeg is installed:

To process a single scan video:
```python main.py --input /path/to/video.mp4 --output ./results/ --method lightglue```
Available methods: orb, mudic, lightglue

## Output
- Aligned 3D intensity matrices
- Composite images (max and min)
- Logs of transformation parameters and evaluation scores

## Reference

If you use this work, please cite:

> Leclercq, E.P. (2025). High-resolution Registration of Irradiated Concrete Scans [Master’s thesis]. Charles University.



