# 📌 Realistic Attacks on GUI Agents via Small Trigger Image under Dynamic Visual Contexts

### 📖 Abstract

Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) are increasingly deployed to interact with websites and online services. While these agents offer powerful multimodal reasoning capabilities, their access to open-world content introduces serious security risks, particularly from Environmental Injection Attacks (EIAs), where malicious elements embedded in a webpage can hijack agent behavior. Existing works on EIAs have typically exposed vulnerabilities in GUI agents by manipulating the underlying HTML source code or applying perturbations to entire screenshots. While these approaches are effective, they rely on unrealistic assumptions, often requiring the attacker to have administrative privileges over the website. In this work, we consider a more practical and constrained threat model in which the attacker is an ordinary user who can upload only a single trigger image. This setting reflects a common feature of many interactive platforms but presents substantial challenges: the trigger’s position and visual context vary dynamically, and the trigger image itself typically occupies only a small fraction of the screenshot, limiting its visual prominence. To address these challenges, we propose Chameleon, an attack framework with two main novelties. The first is LLM-Driven Environment Simulation, which automatically generates diverse and high-fidelity webpage simulations together with realistic user instructions, enabling robustness against dynamic visual contexts. The second is Attention Black Hole, which transforms attention weights into explicit supervisory signals that guide the agent’s focus toward the trigger region and mitigate the difficulty caused by its limited visual prominence. We evaluate Chameleon on six realistic websites and four representative LVLM-powered GUI agents, where it significantly outperforms existing methods. For example, the average attack success rate on OS-Atlas-Base-7B increases from 5.26% to 32.60%. Ablation studies confirm that both novelties are critical to performance, and a closed-loop sandbox experiment further demonstrates that Chameleon can successfully hijack agent behavior in conditions that closely mirror real-world usage. Our findings reveal underexplored vulnerabilities in modern GUI agents and establish a robust foundation for future research on defense in open-world GUI agent systems.

![overview](./assets/overview.png)

---

### 🚀 Features

* ✅ **Realistic Threat Model**: Assumes attacker is a regular user; can only upload a trigger image.
* 🔄 **Dynamic Layout Robustness**: Supports attack generalization under changing trigger positions and surrounding contexts.
* 🎯 **Small Trigger Effectiveness**: Remains effective even when occupying <10% of the webpage.
* 🧠 **LLM-Driven Environment Simulation**: Leverages LLMs to generate diverse, realistic webpages and user instructions.
* 🕳️ **Attention Black Hole**: Uses attention supervision to ensure model consistently focuses on the trigger.
* 🧪 **Extensive Evaluation**: Supports 4 GUI agents (UI-TARS, OS-Atlas, Qwen2-VL, LLaVA-1.5) and 6 realistic websites.

---

### 🛠️ Requirements

* Python >= 3.9
* PyTorch >= 2.0

We provide **two separate `requirements.txt` files** to support different LVLM backends:

* `requirements_llava.txt`: Required dependencies for running **LLaVA-based models**.
* `requirements_qwen.txt`: Required dependencies for running **Qwen-series models** (e.g., Qwen2-VL).

You can install the corresponding environment with:

```bash
# For LLaVA-1.5-13B
pip install -r requirements_llava.txt

# For Qwen2-VL-7B、OS-Atlas-Base-7B、UI-TARS-7B-DPO
pip install -r requirements_qwen.txt
```

> 💡 Make sure to use a clean virtual environment for each configuration if you're switching between backends.

---

### 🖼️ Model Weights

You will need the following models:

| Model            | Family      | Download Link / Instruction        |
| ---------------- | --------- | ---------------------------------- |
| UI-TARS-7B-DPO   | Qwen      | https://huggingface.co/ByteDance-Seed/UI-TARS-7B-DPO |
| OS-Atlas-Base-7B | Qwen      | https://huggingface.co/OS-Copilot/OS-Atlas-Base-7B            |
| Qwen2-VL-7B      | Qwen      | https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct      |
| LLaVA-1.5-13B    | LLaVa     | https://huggingface.co/liuhaotian/llava-v1.5-13b         |

---

### 📦 Dataset

We take Amazon as an example target website. For the datasets used in training, validation, and testing, we provide several representative samples under the `dataset/` directory. In addition, users can generate training, validation, and test datasets for any website by leveraging the dataset construction code provided in the `LSE/` module.

---

### 🧩 Usage Guide

* The `train/` directory contains the code for **optimizing the trigger image**.
* The `LSE/` directory includes the full implementation of **LLM-Driven Environment Simulation**.
* The `infer/` directory provides code for **applying the optimized trigger to attack GUI agents**.
* The `eval/` directory contains scripts for **evaluating attack effectiveness**, including ASR computation.

---
