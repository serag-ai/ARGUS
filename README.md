# 🧠 ARGUS: Agentic Reasoning and General Understanding System with Applications to Medical Image Analysis
### A Multi-Agent Framework for Multimodal Medical Image Analysis

## 📌 Overview

**ARGUS** is an **agentic AI framework** designed for **multimodal medical image analysis**, integrating multiple specialized agents within a unified, orchestrated system.

Unlike traditional static pipelines, ARGUS dynamically constructs workflows based on:

- User queries  
- Input modality (MRI, pathology, OCT)  
- Intermediate outputs and validation  

At its core, ARGUS employs an **Orchestrator Agent** that performs reasoning and tool selection using a **ReAct-style paradigm**, enabling adaptive, context-aware execution across complex clinical tasks.

---

## 🧩 Key Features

- 🧠 **Agentic Reasoning (ReAct-based Orchestration)**
- 🔄 **Dynamic Workflow Construction**
- 🏥 **Multimodal Support**
  - Radiology (MRI)
  - Pathology (Hematology)
  - Ophthalmology (OCT)
- ✅ **Verification Agent for Quality Control**
- 📊 **Quantification Agents**
  - Volume estimation
  - Cell counting
  - Retinal thickness
- 📚 **Knowledge Retrieval Integration (Literature-aware)**
- 📝 **Automated Clinical Report Generation**

---

## 🏗️ Framework Architecture

ARGUS is composed of modular agents coordinated by a central orchestrator:

### Core Agents

| Agent | Function |
|------|--------|
| **Orchestrator Agent** | Interprets queries, plans workflows, coordinates execution |
| **Processing Agents** | Perform modality-specific segmentation |
| **Verification Agent** | Validates outputs before downstream processing |
| **Quantification Agents** | Extract clinical measurements |
| **Knowledge Retrieval Agent** | Provides literature-based context |
| **Report Agent** | Generates structured clinical reports |

This modular design enables **scalability, extensibility, and robustness**.

---

## 🔬 Supported Pipelines

### 🧠 MRI (Radiology)
- Brain segmentation
- Volumetric analysis (e.g., hippocampus, thalamus)
- Validation of anatomical structures

### 🧪 Pathology
- Cell detection via heatmaps
- Connected component-based cell counting


### 👁️ OCT (Ophthalmology)
- Retinal layer segmentation
- Thickness estimation (RNFL, GCL/IPL, RPE)


---

## 📊 Datasets

The framework was evaluated using publicly available datasets:

- IXI Dataset (MRI)
- ALL-IDB1 Dataset (Pathology)
- OCT Dataset (Retinal Imaging)

📌 **Note:**  
All datasets are **publicly accessible through their official sources**.

- **IXI Dataset (MRI)**  
  https://brain-development.org/ixi-dataset/

- **ALL-IDB1 Dataset (Pathology)**  
  https://scotti.di.unimi.it/all/

- **OCT Dataset (Retinal Imaging)**  
  https://doi.org/10.1364/BOE.420456  
  *(Available upon request from original authors)*

## ⚙️ Model Weights

Pretrained model weights are available per modality:

- MRI Segmentation (QuickNAT): *Available via original QuickNAT implementation*
- Pathology Segmentation Model: *(provide upon request due to its large volume)*
- OCT Segmentation Model: *(provide upon request due to its large volume)*

---

## 🙏 Acknowledgment

### QuickNAT

The MRI segmentation agent is based on:

> **QuickNAT: A Fully Convolutional Network for Fast and Accurate Brain Segmentation**

We acknowledge and credit the original authors for their contribution to medical image segmentation.

---

## 🧪 Evaluation

The framework is evaluated across multiple dimensions:

- ✅ Workflow completion rate  
- 📊 Quantitative accuracy (Bland–Altman analysis)  
- 🧠 Reasoning and knowledge integration  
- ⚠️ Robustness and failure handling  

### Key Observations

- High reliability across all modalities  
- Safe early termination when validation fails  
- Strong agreement with ground truth measurements  
- Zero unsafe predictions in pathology evaluation  
- Effective human-in-the-loop escalation for uncertain cases  

---

## ⚡ Performance

Average processing time per case (NVIDIA T4 GPU):

| Modality | Avg Time / Case |
|---------|---------------|
| MRI | ~31.95 sec |
| Pathology | ~24.91 sec |
| OCT | ~44.41 sec |

## 🔍 Example Query

```python
query = "Compute hippocampus volume and compare with normal range"

response = orchestrator.run(
    query=query,
    input_path="IMAGE"
)
