

<h2 align="center"> <a href="">CrashChat: A Multimodal Large Language Model for Multitask Traffic Crash Video Analysis</a></h2>

<h4 align="center"> <a href="https://github.com/Liangkd/">Kaidi Liang</a>, <a href="">Ke Li</a>, <a href="">Xianbiao Hu</a>, <a href="">Ruwen Qin*</a></h4>

<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2312.02051'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
<a href='https://huggingface.co/datasets/KDliang/CrashChat'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a> 
<a href='https://huggingface.co/KDliang/crashchat'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a> 
</div>


## 📰 News
- [25.12.18] ![NEW!](https://img.shields.io/badge/NEW!-red) Release the initial version of **CrashChat**.


## 🌟 Introduction
- **CrashChat** is a Multimodal Large Language Model specifically designed for traffic crash video analysis. Our model incorporates three key architectural contributions: 
  - (1) a multitask learning approach designed to effectively inject the comprehensive knowledge of crash video analysis into VideoLLaMA3;
  - (2) an MLLM capable of unified crash recognition, temporal grounding, and understanding across diverse scenarios;
  - (3) a comprehensive evaluation that provides the first benchmarking of MLLMs for end-to-end crash video analysis.
- We also construct an instruction-tuning crash video dataset, encompassing six core tasks and a total of 18,385 videos and 96,184 video-QA pairs, to further enhance CrashChat's performance.

<p align="center" width="100%">
<a target="_blank"><img src="figs/figs/model_architecture_overview.png" alt="CrashChat" style="width: 60%; min-width: 400px; display: block; margin: auto;"></a>
</p>


## 🤗 Example Outputs
- **An illustration of linguistic-centric task and perception-centric task of CrashChat**

<p align="center" width="100%">
<a target="_blank"><img src="figs/Qualitative_analysis.png" alt="Qualitative_analysis" style="width: 60%; min-width: 400px; display: block; margin: auto;"></a>
</p>


<details>
  <summary>💡Click here to show detailed performance on current baseline MLLM</summary>
  <img src="figs/ComparisonofMLLMsforchoosingthebaselinemodel..png" style="max-width: 100%; height: auto;">
</details>

<details>
  <summary>💡Click here to show detailed performance on coupling and grouping strategy for both linguistic-centric and perception-centric tasks</summary>
  <img src="figs/ComparisonofMLLMsforchoosingthebaselinemodel..png" style="max-width: 100%; height: auto;">
</details>




## 🤖 Model Zoo (Fine-tuned Checkpoints)

The following checkpoints store learnable parameters (positional linear projection layers, and lora) only.


| Checkpoint | MLLM backbone | Training Strategy - Task | Link |
|----------|-------------|-------------|--------------|
| VideoLLaMA-3 | VideoLLaMA-3 7B | VideoLLaMA3 - baseline | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/videollama3_baseline) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Independent monotask models - crash recognition | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/crash_recognition_independent_monotask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Independent monotask models - crash description | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/crash_description_independent_monotask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Independent monotask models - causal reasoning | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/causal_reasoning_independent_monotask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Independent monotask models - prevention reasoning | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/prevention_reasoning_independent_monotask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Independent monotask models - pre-crash localization | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/pre_crash_localization_independent_monotask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Independent monotask models - crash localization | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/crash_localization_independent_monotask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Homogeneous multitask models - linguistic-centric tasks | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/linguistic_centric_homogeneous_multitask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Homogeneous multitask models - perception-centric tasks | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/perception_centric_homogeneous_multitask) |
| CrashChat-7B-Finetuned | VideoLLaMA-3 7B | Heterogeneous multitask models - all tasks | [Weights](https://huggingface.co/KDliang/crashchat/tree/main/ckpt/heterogeneous_multitask) |


**Notes:**
- Fine-tuned on instruction-tuning data from  
  - [CrashChat-original_01](https://huggingface.co/datasets/KDliang/CrashChat/tree/main/CrashChat-original)
  - [CrashChat-resized_01](https://huggingface.co/datasets/KDliang/CrashChat/tree/main/CrashChat-resized)
  - [CrashChat-resized_02](https://huggingface.co/datasets/KDliang/CrashChat/tree/main/CrashChat-resized_02)



## 🛠️ Requirements and Installation

#### Enviroment Preparation 

Step 1: Create and activate a Conda environment:
```
conda create -n crashchat python=3.10 -y
conda activate crashchat
```

Step 2, Upgrade pip and install PyTorch (CUDA 11.8):
```
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu118
```

Step 3, Install required Python dependencies:
```
pip install -r requirements.txt
```

Step 4, Install FlashAttention (local wheel). You can download the cooresponding wheel [here](https://huggingface.co/KDliang/crashchat/tree/main/wheels):
```
pip install /CrashChat/flash_attn-2.7.3+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-deps
```

Step 5, Install FFmpeg:
```
conda install -c conda-forge ffmpeg -y
```

#### Prerequisites 

To use our training code, please organize checkpoint as you like under `ckpt`, and then download the corresponding model checkpoint from Model Zoo. For example:
```bash
ckpt
├── videollama3_baseline (required for training)
│   ├── added_tokens.json
│   └── ...
├── crash_recognition_independent_monotask (optional for evaluation)
│   ├── added_tokens.json
│   └── ...
├── crash_description_independent_monotask (optional for evaluation)
│   ├── added_tokens.json
│   └── ...
├── causal_reasoning_independent_monotask (optional for evaluation)
├── prevention_reasoning_independent_monotask (optional for evaluation)
├── pre_crash_localization_independent_monotask (optional for evaluation)
├── crash_localization_independent_monotask (optional for evaluation)
├── linguistic_centric_homogeneous_multitask (optional for evaluation)
├── perception_centric_homogeneous_multitask (optional for evaluation)
└── heterogeneous_multitask (optional for evaluation)
```

To use our training code, please organize the video data and annotations files as you like under `data`. For example:
```bash
data
├── videos
│   ├── cap_1_001537.mp4
│   ├── cap_1_002004.mp4
│   └── ...
├── crashchat_dada_video_total_cause_reasoning_test.json
├── crashchat_dada_video_total_cause_reasoning_train.json
├── ...
└── crashchat_dada_video_total_prevention_reasoning_val.json
```


## 🗝️ Training








