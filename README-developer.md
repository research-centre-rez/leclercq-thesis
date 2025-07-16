# Developer README

Welcome, contributor! This guide explains the internal structure and technical details of the thesis codebase for high-resolution video registration of irradiated concrete samples.

## Project Overview

The system ingests 4K videos of rotating concrete samples and produces aligned 3D intensity matrices and fused images. It’s built as a modular pipeline allowing experimentation with multiple registration techniques under varying lighting conditions.

### Pipeline Stages

1. **Video Preprocessing**: Cutting videos into light-specific sections.
2. **Rotation Correction**: Sparse optical flow to "unrotate" frames.
3. **Image Registration**:
   - `µDIC`: area-based
   - `ORB`: feature-based
   - `LightGlue`: DL-based (with SuperPoint)
4. **Image Fusion**: Min/max pixel fusion
5. **Evaluation**: Sharpness metrics (NGLV, Brenner)

## Repository Layout

## Dependencies

## Benchmarks
% tady by mělo být stručné overview co je v práci a link na ní

## Planned Improvements
% seznam toho co se nestihlo

## Maintainers

Erik Philippe Leclercq (Original author)
Research Centre Řež – Honza Blažek (Supervisor) - Imaging and Materials Lab 
