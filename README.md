# 📊 Benchmarking Self-Supervised Learning for Denoising Voltage Imaging Data

This repository contains code and model checkpoints used to benchmark self-supervised denoising models on two types of voltage imaging datasets:

- 🧪 **Synthetic dataset**: *Optosynth*  
- 🧠 **Real dataset**: *HPC2*

---

## 📁 Folder Structure

Each folder includes:

- ✅ **Final trained models**  
  Used for predictions on the Optosynth and HPC2 datasets.

- 🔧 **Modified code**  
  Adapted from the original implementations to support voltage imaging data:
  
  - **Noise2Void**
    - Custom training and prediction scripts
    - tSNR (temporal Signal-to-Noise Ratio) calculation code
    
  - **AP-BSN**
    - Adapted data loaders
    - Configuration files for training and inference

  For more information on how to use the original models, check out their respective repositories on Github
---
