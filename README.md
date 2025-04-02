# Enhancing the UNet3+ Architecture for Deep Learning Segmentation of Kidneys and Cysts in Autosomal Dominant Polycystic Kidney Disease (ADPKD)

This repository contains the **official code implementation** of our SPIE Medical Imaging paper:

**[Enhancing the UNet3+ architecture for deep learning segmentation of kidneys and cysts in Autosomal Dominant Polycystic Kidney Disease (ADPKD)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12930/3006823/Enhancing-the-UNet3-architecture-for-deep-learning-segmentation-of-kidneys/10.1117/12.3006823.short)**  
📄 *Presented at SPIE Medical Imaging 2024*

---

## 📌 Overview

This work presents a modified version of the **UNet3+ architecture** tailored for the complex task of **segmenting kidneys and cysts** in patients with **Autosomal Dominant Polycystic Kidney Disease (ADPKD)**.

### 🔧 Key Innovation

We introduce a novel **concatenated skip connection strategy**, which improves upon traditional skip connections by:

- Preserving high-frequency spatial details critical for **boundary-level segmentation**
- Capturing multi-scale features more effectively across layers
- Enhancing segmentation accuracy for **both kidneys and cysts**, especially in large, heterogeneous ADPKD cases

---

## 🔄 Related Works

This repository also partially supports the codebase for the following articles:

- **[Deep Transfer Learning from Constrained Source to Target Domains in Medical Image Segmentation](https://library.imaging.org/jist/articles/68/6/060505)**  
  *Published in Journal of Imaging Science and Technology (JIST), 2024*

- **[Deep transfer learning from limited source for abdominal CT and MR image segmentation](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12926/129262T/Deep-transfer-learning-from-limited-source-for-abdominal-CT-and/10.1117/12.3006814.short)**  
  *Presented at SPIE Medical Imaging 2024*

> 🧠 The modified UNet3+ backbone used in this repository serves as a foundational model for transfer learning in medical image segmentation across multiple modalities and domains.

---

## 📚 Dataset

This project uses the **CRISP dataset**  
(Consortium for Radiologic Imaging Studies of Polycystic Kidney Disease) — a longitudinal collection of MRI scans for patients with **ADPKD**.

📎 Dataset access: [https://repository.niddk.nih.gov/study/10](https://repository.niddk.nih.gov/study/10)

### 🧪 To Reproduce Our Results

Use the following notebook provided in this repository:

```
(1) Raw_Image_Processing.ipynb
```

This notebook handles:

- 3D to 2D slice conversion  
- Left/right kidney cropping  
- Input-label preparation (kidney and cyst masks)  
- Data formatting for model training

> ⚠️ Make sure your input data follows the same structure and naming conventions described in the notebook for correct preprocessing.

---

## 🛠️ Requirements

This project is built using **Python** and **TensorFlow** for deep learning-based segmentation of kidneys and cysts. GPU acceleration is supported for faster training and inference.

### 📦 Core Dependencies

- `python >= 3.7`
- `tensorflow-gpu >= 2.6.0`
- `nibabel` *(for NIfTI/medical imaging formats)*
- `numpy`
- `matplotlib`
- `opencv-python`
- `scikit-image`
- `scipy`
- `pandas`
- `tqdm`

### ✅ Installation

Install all required dependencies using pip:

```bash
pip install tensorflow-gpu nibabel numpy matplotlib opencv-python scikit-image scipy pandas tqdm
```

> ⚠️ Make sure your system has a compatible **CUDA** and **cuDNN** version installed to support `tensorflow-gpu`. Refer to the [TensorFlow GPU setup guide](https://www.tensorflow.org/install/gpu) for installation help.

> 💡 For best results, use a virtual environment (`venv` or `conda`) to manage packages.

---

## ▶️ How to Run

The workflow is organized using a **step-by-step notebook structure**, designed for easy execution and reproducibility.

### 📓 Notebooks (Run in Order)

1. **`(1) Raw Image Processing.ipynb`**  
   Prepares the CRISP dataset (3D to 2D slicing, cropping, formatting)

2. **`(2) Train and Prediction.ipynb`**  
   Trains the enhanced UNet3+ model and performs inference

3. **`(3) Performance Calculation.ipynb`**  
   Computes segmentation metrics like Dice score and visual comparisons

### 🧠 Model Code

The `Networks/` folder contains the **model architecture and core utilities**, including:

- UNet3+ model with enhanced concatenated skip connections
- Loss functions (Dice, combined losses)
- Utility layers, encoders/decoders, and attention mechanisms

### 🧰 Additional Scripts

- `Data_Gen_2D.py`: Custom data generator for 2D input batches  
- `arch_parts.py`: Contains model submodules and reusable architectural blocks  

> ✅ Follow the notebook order, and ensure all paths to data and checkpoints are properly set before running training or evaluation.

---

## 🧠 Model Compatibility & Weights

The proposed **Concatenated Skip Connections (CSCs)** are modular and can be easily integrated into any **UNet-based architecture** that uses traditional skip connections.

### 🔄 Compatibility

- Works seamlessly with standard **UNet**, **UNet++**, and **UNet3+** backbones
- Can be applied to both 2D and 3D medical image segmentation models
- Enhances the ability to preserve fine-grained spatial information during upsampling

### 💾 Pretrained Weights

Pretrained model weights are **available upon request**.  
Please contact the authors or open an issue on this repository for access.

> 📬 We share weights for **non-commercial research purposes** only.

---

## 📝 Citation

If you use this code or build upon our work, please cite the following papers:

1. **Deep Transfer Learning from Limited Source for Abdominal CT and MR Image Segmentation**  
   Chetana Krishnan, Emma Schmidt, Ezinwanne Onuoha, Michal Mrug, Carlos E. Cardenas, Harrison Kim  
   *SPIE Medical Imaging 2024: Image Processing*  
   [https://doi.org/10.1117/12.3006814](https://doi.org/10.1117/12.3006814)

2. **Deep Transfer Learning from Constrained Source to Target Domains in Medical Image Segmentation**  
   Chetana Krishnan, Emma Schmidt, Ezinwanne Onuoha, Sean Mullen, Ronald Roye, Phillip Chumley, Michal Mrug, Carlos E. Cardenas, Harrison Kim,  
   Consortium for Radiologic Imaging Studies of Polycystic Kidney Disease (CRISP) investigators  
   *Journal of Imaging Science and Technology, 2024, pp. 1–10*  
   [https://doi.org/10.2352/J.ImagingSci.Technol.2024.68.6.060505](https://doi.org/10.2352/J.ImagingSci.Technol.2024.68.6.060505)

3. **Enhancing the UNet3+ Architecture for Deep Learning Segmentation of Kidneys and Cysts in Autosomal Dominant Polycystic Kidney Disease (ADPKD)**  
   Chetana Krishnan, Emma Schmidt, Ezinwanne Onuoha, Michal Mrug, Carlos E. Cardenas, Harrison Kim  
   *SPIE Medical Imaging 2024: Clinical and Biomedical Imaging*  
   [https://doi.org/10.1117/12.3006823](https://doi.org/10.1117/12.3006823)

> 📬 Please reference these works in any derivative research or publications.
---

## 📄 License

This project is licensed under the **MIT License**.

