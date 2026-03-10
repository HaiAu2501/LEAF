<div align="center">

# 🍁 MAPLE
### Multi-Agent Prior Learning for Constructing Tree Ensembles

[![ICLR 2026 Workshop](https://img.shields.io/badge/ICLR%202026-MALGAI%20Workshop-blue?style=flat-square&logo=academia)](https://openreview.net/forum?id=bBIA4TVEBG)
[![OpenReview](https://img.shields.io/badge/OpenReview-bBIA4TVEBG-8C1515?style=flat-square&logo=openreview)](https://openreview.net/forum?id=bBIA4TVEBG)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Better trees. Better diversity. Better performance.*

</div>

---

## 📖 Overview

Tree ensembles remain among the strongest methods for tabular data — yet they typically sample features **uniformly**, ignoring task-specific knowledge. **MAPLE** changes this by integrating LLM-derived feature priors directly into ensemble construction through a multi-agent framework.

Multiple LLM agents — each with a distinct perspective *(domain expert, statistician, conservative, exploratory)* — propose feature importance priors. A **bandit algorithm** adaptively allocates trees across agents based on out-of-bag performance, while **hardness-aware rewards** ensure the ensemble focuses on difficult examples.

---

## ✨ Key Ideas

| | Feature | Description |
|---|---|---|
| 🤖 | **LLM-Guided Feature Priors** | Each agent queries an LLM (GPT-4o-mini) with dataset metadata to obtain a prior distribution over features. Trees sample features from these priors instead of uniformly, focusing splits on the most promising dimensions. |
| 🌐 | **Multi-Agent Diversity** | Different agents encode different inductive biases. A UCB1 bandit selects which agent generates each tree, balancing exploitation of strong agents with exploration of diverse perspectives. |
| 🔄 | **Adaptive Prior Updates** | Feature priors evolve during ensemble construction. Features that contribute to strong performance on hard samples receive higher weight; overused features are penalized to maintain diversity. |

---

## 🚀 Quick Start

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Create a `.env` file with your OpenAI API key:**

```env
OPENAI_API_KEY=your_api_key_here
```

**3. Run the main script:**

```bash
python main.py
```

**4. Try different datasets and configurations** by overriding hyperparameters:

```bash
python main.py \
       alg=rf_cls \
       dataset.name=compas \
       evaluator.model=gpt-5-mini
```

---

## 📝 Citation

If you find MAPLE useful, please cite our work:

```bibtex
@inproceedings{kiet2026maple,
    title={{MAPLE}: Multi-Agent Prior Learning for Constructing Tree Ensembles},
    author={Nguyen Viet Tuan Kiet and Nguyen Ba Thinh and Thanh Trung Huynh and Hieu Pham},
    booktitle={Workshop on Multi-Agent Learning and Its Opportunities in the Era of Generative AI},
    year={2026},
    url={https://openreview.net/forum?id=bBIA4TVEBG}
}
```
