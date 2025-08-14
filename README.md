# Awesome-Video-LLM-Post-Training [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This Awesome list systematically curates and tracks the latest research in the post-training of Video-LLMs, with a special emphasis on works that enhance their reasoning capabilities. Following the taxonomy of the field, we focus on three key paradigms:

- 🧠 Reinforced Video-LLMs: Exploring how RL techniques are used to align Video-LLMs with human preferences or specific metrics. This includes methods like RLHF, DPO, GRPO and the design of effective reward models to enhance the logical consistency and factuality of model outputs.

- ⚙️ SFT for Reasoning: Collecting studies that leverage SFT on meticulously curated, reasoning-centric datasets. These works often incorporate CoT or other structured formats to directly teach models how to perform complex, multi-step reasoning.

- 🚀 Test-Time Scaling in Video Reasoning: Focusing on strategies that enhance reasoning capabilities at inference time without requiring further model training. This includes techniques like agentic frameworks, tool use, RAG, long CoT, and other methods that scale reasoning through computation.

📊 Benchmarks for Video Reasoning: Including the latest and most challenging benchmarks designed specifically to evaluate the complex reasoning abilities of Video-LLMs.

We hope this repository serves as a comprehensive and up-to-date resource hub for researchers and developers in this cutting-edge field. Contributions from the community are highly welcome via Pull Requests!

## Table of Contents

- [Awesome-Video-LLM-Post-Training ](#awesome-video-llm-post-training-)
  - [Table of Contents](#table-of-contents)
  - [Survey](#survey)
    - [Reinforced Video-LLMs](#reinforced-video-llms)
    - [Video-LLM SFT for Reasoning](#video-llm-sft-for-reasoning)
    - [Test-Time Scaling in Video Reasoning](#test-time-scaling-in-video-reasoning)
    - [Benchmarks for Video Reasoning](#benchmarks-for-video-reasoning)
    - [Related Surveys](#related-surveys)
  - [🌟 Star History](#-star-history)
  - [📝 Citation](#-citation)

## Survey

### Reinforced Video-LLMs

| Title | Paper | Code | Dataset | Venue |
| :--- | :---: | :---: | :---: | :---: |
| Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning | [Link](https://arxiv.org/abs/2508.04416) | | | |
| ARC-Hunyuan-Video-7B: Structured Video Comprehension of Real-World Shorts | [Link](https://arxiv.org/abs/2507.20939) | | | |
| EmbRACE-3K: Embodied Reasoning and Action in Complex Environments | [Link](https://arxiv.org/abs/2507.10548) | | | |
| Scaling RL to Long Videos | [Link](https://arxiv.org/abs/2507.07966) | | | |
| Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning | [Link](https://arxiv.org/abs/2507.06485) | | | |
| Tempo-R0: A Video-MLLM for Temporal Video Grounding through Efficient Temporal Sensing Reinforcement Learning | [Link](https://arxiv.org/abs/2507.04702) | | | |
| VRAgent-R1: Boosting Video Recommendation with MLLM-based Agents via Reinforcement Learning | [Link](https://arxiv.org/abs/2507.02626) | | | |
| Kwai Keye-VL Technical Report | [Link](https://arxiv.org/abs/2507.01949) | | | |
| VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning | [Link](https://arxiv.org/abs/2506.17221) | | | |
| Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning | [Link](https://arxiv.org/abs/2506.13654) | | | |
| VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks | [Link](https://arxiv.org/abs/2506.09079) | | | |
| DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO | [Link](https://arxiv.org/abs/2506.07464) | | | |
| AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs | [Link](https://arxiv.org/abs/2506.05328) | | | |
| MiMo-VL Technical Report | [Link](https://arxiv.org/abs/2506.03569) | | | |
| EgoVLM: Policy Optimization for Egocentric Video Understanding | [Link](https://arxiv.org/abs/2506.03097) | | | |
| Reinforcement Learning Tuning for VideoLLMs: Reward Design and Data Efficiency | [Link](https://arxiv.org/abs/2506.01908) | | | |
| VideoCap-R1: Enhancing MLLMs for Video Captioning via Structured Thinking | [Link](https://arxiv.org/abs/2506.01725) | | | |
| ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding | [Link](https://arxiv.org/abs/2506.01300) | | | |
| ReFoCUS: Reinforcement-guided Frame Optimization for Contextual Understanding | [Link](https://arxiv.org/abs/2506.01274) | | | |
| Reinforcing Video Reasoning with Focused Thinking | [Link](https://arxiv.org/abs/2505.24718) | | | |
| VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning | [Link](https://arxiv.org/abs/2505.23504) | | | |
| A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding | [Link](https://arxiv.org/abs/2505.21962) | | | |
| MUSEG: Reinforcing Video Temporal Understanding via Timestamp-Aware Multi-Segment Grounding | [Link](https://arxiv.org/abs/2505.20715) | | | |
| Omni-R1: Reinforcement Learning for Omnimodal Reasoning via Two-System Collaboration | [Link](https://arxiv.org/abs/2505.20256) | | | |
| Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought | [Link](https://arxiv.org/abs/2505.19877) | | | |
| VerIPO: Cultivating Long Reasoning in Video-LLMs via Verifier-Gudied Iterative Policy Optimization | [Link](https://arxiv.org/abs/2505.19000) | | | |
| Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning | [Link](https://arxiv.org/abs/2505.16836) | | | |
| From Evaluation to Defense: Advancing Safety in Video Large Language Models | [Link](https://arxiv.org/abs/2505.16643) | | | |
| Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning | [Link](https://arxiv.org/abs/2505.15966) | | | |
| ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning | [Link](https://arxiv.org/abs/2505.15447) | | | |
| UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning | [Link](https://arxiv.org/abs/2505.14231) | | | |
| BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation | [Link](https://arxiv.org/abs/2505.12620) | | | |
| VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning | [Link](https://arxiv.org/abs/2505.12434) | | | |
| Seed1.5-VL Technical Report | [Link](https://arxiv.org/abs/2505.07062) | | | |
| Compile Scene Graphs with Reinforcement Learning | [Link](https://arxiv.org/abs/2504.13617) | | | |
| Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization | [Link](https://arxiv.org/abs/2504.12083) | | | |
| Mavors: Multi-granularity Video Representation for Multimodal Large Language Model | [Link](https://arxiv.org/abs/2504.10068) | | | |
| TinyLLaVA-Video-R1: Towards Smaller LMMs for Video Reasoning | [Link](https://arxiv.org/abs/2504.09641) | | | |
| VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning | [Link](https://arxiv.org/abs/2504.06958) | | | |
| Spatial-R1: Enhancing MLLMs in Video Spatial Reasoning | [Link](https://arxiv.org/abs/2504.01805) | | | |
| Improved Visual-Spatial Reasoning via R1-Zero-Like Training | [Link](https://arxiv.org/abs/2504.00883) | | | |
| Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 | [Link](https://arxiv.org/abs/2503.24376) | | | |
| Video-R1: Reinforcing Video Reasoning in MLLMs | [Link](https://arxiv.org/abs/2503.21776) | | | |
| Exploring Hallucination of Large Multimodal Models in Video Understanding: Benchmark, Analysis and Mitigation | [Link](https://arxiv.org/abs/2503.19622) | | | |
| TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM | [Link](https://arxiv.org/abs/2503.13377) | | | |
| ST-Think: How Multimodal Large Language Models Reason About 4D Worlds from Ego-Centric Videos | [Link](https://arxiv.org/abs/2503.12542) | | | |
| Memory-enhanced Retrieval Augmentation for Long Video Understanding | [Link](https://arxiv.org/abs/2503.09149) | | | |
| video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model | [Link](https://arxiv.org/abs/2502.11775) | | | |
| Unhackable Temporal Rewarding for Scalable Video MLLMs | [Link](https://arxiv.org/abs/2502.12081) | | | |
| Temporal Preference Optimization for Long-Form Video Understanding | [Link](https://arxiv.org/abs/2501.13919) | | | |
| InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model | [Link](https://arxiv.org/abs/2501.12368) | | | |
| VidChain: Chain-of-Tasks with Metric-based Direct Preference Optimization for Dense Video Captioning | [Link](https://arxiv.org/abs/2501.06761) | | | |
| VideoSAVi: Self-Aligned Video Language Models without Human Supervision | [Link](https://arxiv.org/abs/2412.00624) | | | |


### Video-LLM SFT for Reasoning

| Title | Paper | Code | Dataset | Venue |
| :--- | :---: | :---: | :---: | :---: |
| Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning | [Link](https://arxiv.org/abs/2508.04416) | | | |
| ARC-Hunyuan-Video-7B: Structured Video Comprehension of Real-World Shorts | [Link](https://arxiv.org/abs/2507.20939) | | | |
| CoTasks: Chain-of-Thought based Video Instruction Tuning Tasks | [Link](https://arxiv.org/abs/2507.13609) | | | |
| EmbRACE-3K: Embodied Reasoning and Action in Complex Environments | [Link](https://arxiv.org/abs/2507.10548) | | | |
| Scaling RL to Long Videos | [Link](https://arxiv.org/abs/2507.07966) | | | |
| Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models | [Link](https://arxiv.org/abs/2507.05822) | | | |
| Kwai Keye-VL Technical Report | [Link](https://arxiv.org/abs/2507.01949) | | | |
| VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning | [Link](https://arxiv.org/abs/2506.17221) | | | |
| Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning | [Link](https://arxiv.org/abs/2506.13654) | | | |
| DAVID-XR1: Detecting AI-Generated Videos with Explainable Reasoning | [Link](https://arxiv.org/abs/2506.14827) | | | |
| VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks | [Link](https://arxiv.org/abs/2506.09079) | | | |
| AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs | [Link](https://arxiv.org/abs/2506.05328) | | | |
| Video-Skill-CoT: Skill-based Chain-of-Thoughts for Domain-Adaptive Video Reasoning | [Link](https://arxiv.org/abs/2506.03525) | | | |
| MiMo-VL Technical Report | [Link](https://arxiv.org/abs/2506.03569) | | | |
| ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding | [Link](https://arxiv.org/abs/2506.01300) | | | |
| Chain-of-Frames: Advancing Video Understanding in Multimodal LLMs via Frame-Aware Reasoning | [Link](https://arxiv.org/abs/2506.00318) | | | |
| Universal Visuo-Tactile Video Understanding for Embodied Interaction | [Link](https://arxiv.org/abs/2505.22566) | | | |
| Fostering Video Reasoning via Next-Event Prediction | [Link](https://arxiv.org/abs/2505.22457) | | | |
| A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding | [Link](https://arxiv.org/abs/2505.21962) | | | |
| Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought | [Link](https://arxiv.org/abs/2505.19877) | | | |
| Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning | [Link](https://arxiv.org/abs/2505.16836) | | | |
| From Evaluation to Defense: Advancing Safety in Video Large Language Models | [Link](https://arxiv.org/abs/2505.16643) | | | |
| Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning | [Link](https://arxiv.org/abs/2505.15966) | | | |
| UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning | [Link](https://arxiv.org/abs/2505.14231) | | | |
| VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning | [Link](https://arxiv.org/abs/2505.12434) | | | |
| Seed1.5-VL Technical Report | [Link](https://arxiv.org/abs/2505.07062) | | | |
| TEMPURA: Temporal Event Masked Prediction and Understanding for Reasoning in Action | [Link](https://arxiv.org/abs/2505.01583) | | | |
| VEU-Bench: Towards Comprehensive Understanding of Video Editing | [Link](https://arxiv.org/abs/2504.17828) | | | |
| Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models | [Link](https://arxiv.org/abs/2504.15271) | | | |
| Compile Scene Graphs with Reinforcement Learning | [Link](https://arxiv.org/abs/2504.13617) | | | |
| Mavors: Multi-granularity Video Representation for Multimodal Large Language Model | [Link](https://arxiv.org/abs/2504.10068) | | | |
| LVC: A Lightweight Compression Framework for Enhancing VLMs in Long Video Understanding | [Link](https://arxiv.org/abs/2504.06835) | | | |
| From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models | [Link](https://arxiv.org/abs/2504.06214) | | | |
| Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 | [Link](https://arxiv.org/abs/2503.24376) | | | |
| Video-R1: Reinforcing Video Reasoning in MLLMs | [Link](https://arxiv.org/abs/2503.21776) | | | |
| PAVE: Patching and Adapting Video Large Language Models | [Link](https://arxiv.org/abs/2503.19794) | | | |
| Exploring Hallucination of Large Multimodal Models in Video Understanding: Benchmark, Analysis and Mitigation | [Link](https://arxiv.org/abs/2503.19622) | | | |
| VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning | [Link](https://arxiv.org/abs/2503.13444) | | | |
| ST-Think: How Multimodal Large Language Models Reason About 4D Worlds from Ego-Centric Videos | [Link](https://arxiv.org/abs/2503.12542) | | | |
| TIME: Temporal-sensitive Multi-dimensional Instruction Tuning and Benchmarking for Video-LLMs | [Link](https://arxiv.org/abs/2503.09994) | | | |
| Memory-enhanced Retrieval Augmentation for Long Video Understanding | [Link](https://arxiv.org/abs/2503.09149) | | | |
| UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces | [Link](https://arxiv.org/abs/2503.06157) | | | |
| Token-Efficient Long Video Understanding for Multimodal LLMs | [Link](https://arxiv.org/abs/2503.04130) | | | |
| M-LLM Based Video Frame Selection for Efficient Video Understanding | [Link](https://arxiv.org/abs/2502.19680) | | | |
| video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model | [Link](https://arxiv.org/abs/2502.11775) | | | |
| Unhackable Temporal Rewarding for Scalable Video MLLMs | [Link](https://arxiv.org/abs/2502.12081) | | | |
| Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuray | [Link](https://arxiv.org/abs/2502.05177) | | | |
| InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model | [Link](https://arxiv.org/abs/2501.12368) | | | |
| Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks | [Link](https://arxiv.org/abs/2501.08326) | | | |
| LongViTU: Instruction Tuning for Long-Form Video Understanding | [Link](https://arxiv.org/abs/2501.05037) | | | |
| VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM | [Link](https://arxiv.org/abs/2501.00599) | | | |
| Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces | [Link](https://arxiv.org/abs/2412.14171) | | | |
| Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling | [Link](https://arxiv.org/abs/2412.05271) | | | |
| STEP: Enhancing Video-LLMs' Compositional Reasoning by Spatio-Temporal Graph-guided Self-Training | [Link](https://arxiv.org/abs/2412.00161) | | | |
| ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos | [Link](https://arxiv.org/abs/2411.14901) | | | |
| VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection | [Link](https://arxiv.org/abs/2411.14794) | | | |
| Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models | [Link](https://arxiv.org/abs/2410.03290) | | | |


### Test-Time Scaling in Video Reasoning

| Title | Paper | Code | Dataset | Venue |
| :--- | :---: | :---: | :---: | :---: |
| Thinking With Videos: Multimodal Tool-Augmented Reinforcement Learning for Long Video Reasoning | [Link](https://arxiv.org/abs/2508.04416) | | | |
| VideoForest: Person-Anchored Hierarchical Reasoning for Cross-Video Question Answering | [Link](https://arxiv.org/abs/2508.03039) | | | |
| Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference | [Link](https://arxiv.org/abs/2508.02134) | | | |
| EgoPrune: Efficient Token Pruning for Egomotion Video Reasoning in Embodied Agent | [Link](https://arxiv.org/abs/2507.15428) | | | |
| Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding | [Link](https://arxiv.org/abs/2507.15028) | | | |
| ViTCoT: Video-Text Interleaved Chain-of-Thought for Boosting Video Understanding in Large Language Models | [Link](https://arxiv.org/abs/2507.09876) | | | |
| Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning | [Link](https://arxiv.org/abs/2507.06485) | | | |
| StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling | [Link](https://arxiv.org/abs/2507.05240) | | | |
| VRAgent-R1: Boosting Video Recommendation with MLLM-based Agents via Reinforcement Learning | [Link](https://arxiv.org/abs/2507.02626) | | | |
| DIVE: Deep-search Iterative Video Exploration A Technical Report for the CVRR Challenge at CVPR 2025 | [Link](https://arxiv.org/abs/2506.21891) | | | |
| How Far Can Off-the-Shelf Multimodal Large Language Models Go in Online Episodic Memory Question Answering? | [Link](https://arxiv.org/abs/2506.16450) | | | |
| Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning | [Link](https://arxiv.org/abs/2506.13654) | | | |
| VideoDeepResearch: Long Video Understanding With Agentic Tool Using | [Link](https://arxiv.org/abs/2506.10821) | | | |
| CogStream: Context-guided Streaming Video Question Answering | [Link](https://arxiv.org/abs/2506.10516) | | | |
| Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency | [Link](https://arxiv.org/abs/2506.08343) | | | |
| CyberV: Cybernetics for Test-time Scaling in Video Understanding | [Link](https://arxiv.org/abs/2506.07971) | | | |
| Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs | [Link](https://arxiv.org/abs/2506.07180) | | | |
| VideoChat-A1: Thinking with Long Videos by Chain-of-Shot Reasoning | [Link](https://arxiv.org/abs/2506.06097) | | | |
| MiMo-VL Technical Report | [Link](https://arxiv.org/abs/2506.03569) | | | |
| ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding | [Link](https://arxiv.org/abs/2506.01300) | | | |
| ReFoCUS: Reinforcement-guided Frame Optimization for Contextual Understanding | [Link](https://arxiv.org/abs/2506.01274) | | | |
| SiLVR: A Simple Language-based Video Reasoning Framework | [Link](https://arxiv.org/abs/2505.24869) | | | |
| Multi-RAG: A Multimodal Retrieval-Augmented Generation System for Adaptive Video Understanding | [Link](https://arxiv.org/abs/2505.23990) | | | |
| VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning? | [Link](https://arxiv.org/abs/2505.23359) | | | |
| Omni-R1: Reinforcement Learning for Omnimodal Reasoning via Two-System Collaboration | [Link](https://arxiv.org/abs/2505.20256) | | | |
| Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding | [Link](https://arxiv.org/abs/2505.18079) | | | |
| Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning | [Link](https://arxiv.org/abs/2505.15966) | | | |
| ViQAgent: Zero-Shot Video Question Answering via Agent with Open-Vocabulary Grounding Validation | [Link](https://arxiv.org/abs/2505.15928) | | | |
| ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning | [Link](https://arxiv.org/abs/2505.15447) | | | |
| RVTBench: A Benchmark for Visual Reasoning Tasks | [Link](https://arxiv.org/abs/2505.11838) | | | |
| CoT-Vid: Dynamic Chain-of-Thought Routing with Self Verification for Training-Free Video Reasoning | [Link](https://arxiv.org/abs/2505.11830) | | | |
| VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models | [Link](https://arxiv.org/abs/2505.08455) | | | |
| Seed1.5-VL Technical Report | [Link](https://arxiv.org/abs/2505.07062) | | | |
| Empowering Agentic Video Analytics Systems with Video Language Models | [Link](https://arxiv.org/abs/2505.00254) | | | |
| Divide and Conquer: Exploring Language-centric Tree Reasoning for Video Question-Answering | [Link](https://openreview.net/forum?id=yTpn3QY9Ff) | | | |
| SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding | [Link](https://arxiv.org/abs/2504.21435) | | | |
| VideoMultiAgents: A Multi-Agent Framework for Video Question Answering | [Link](https://arxiv.org/abs/2504.20091) | | | |
| MR. Video: "MapReduce" is the Principle for Long Video Understanding | [Link](https://arxiv.org/abs/2504.16082) | | | |
| Multimodal Long Video Modeling Based on Temporal Dynamic Context | [Link](https://arxiv.org/abs/2504.10443) | | | |
| VideoAgent2: Enhancing the LLM-Based Agent System for Long-Form Video Understanding by Uncertainty-Aware CoT | [Link](https://arxiv.org/abs/2504.04471) | | | |
| WikiVideo: Article Generation from Multiple Videos | [Link](https://arxiv.org/abs/2504.00939) | | | |
| Aurelia: Test-time Reasoning Distillation in Audio-Visual LLMs | [Link](https://arxiv.org/abs/2503.23219) | | | |
| Online Reasoning Video Segmentation with Just-in-Time Digital Twins | [Link](https://arxiv.org/abs/2503.21056) | | | |
| From Trial to Triumph: Advancing Long Video Understanding via Visual Context Sample Scaling and Self-reward Alignment | [Link](https://arxiv.org/abs/2503.20472) | | | |
| Agentic Keyframe Search for Video Question Answering | [Link](https://arxiv.org/abs/2503.16032) | | | |
| VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning | [Link](https://arxiv.org/abs/2503.13444) | | | |
| Does Your Vision-Language Model Get Lost in the Long Video Sampling Dilemma? | [Link](https://arxiv.org/abs/2503.12496) | | | |
| Memory-enhanced Retrieval Augmentation for Long Video Understanding | [Link](https://arxiv.org/abs/2503.09149) | | | |
| Everything Can Be Described in Words: A Simple Unified Multi-Modal Framework with Semantic and Temporal Alignment | [Link](https://arxiv.org/abs/2503.09081) | | | |
| QuoTA: Query-oriented Token Assignment via CoT Query Decouple for Long Video Comprehension | [Link](https://arxiv.org/abs/2503.08689) | | | |
| Token-Efficient Long Video Understanding for Multimodal LLMs | [Link](https://arxiv.org/abs/2503.04130) | | | |
| M-LLM Based Video Frame Selection for Efficient Video Understanding | [Link](https://arxiv.org/abs/2502.19680) | | | |
| TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding | [Link](https://arxiv.org/abs/2502.19400) | | | |
| CoS: Chain-of-Shot Prompting for Long Video Understanding | [Link](https://arxiv.org/abs/2502.06428) | | | |
| Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuray | [Link](https://arxiv.org/abs/2502.05177) | | | |
| Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge | [Link](https://arxiv.org/abs/2501.13468) | | | |
| InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model | [Link](https://arxiv.org/abs/2501.12368) | | | |
| The Devil is in Temporal Token: High Quality Video Reasoning Segmentation | [Link](https://arxiv.org/abs/2501.08549) | | | |
| MECD+: Unlocking Event-Level Causal Graph Discovery for Video Reasoning | [Link](https://arxiv.org/abs/2501.07227) | | | |
| VidChain: Chain-of-Tasks with Metric-based Direct Preference Optimization for Dense Video Captioning | [Link](https://arxiv.org/abs/2501.06761) | | | |
| Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs | [Link](https://arxiv.org/abs/2501.04336) | | | |
| Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding | [Link](https://arxiv.org/abs/2501.00358) | | | |
| PruneVid: Visual Token Pruning for Efficient Video Large Language Models | [Link](https://arxiv.org/abs/2412.16117) | | | |
| Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces | [Link](https://arxiv.org/abs/2412.14171) | | | |
| VCA: Video Curious Agent for Long Video Understanding | [Link](https://arxiv.org/abs/2412.10471) | | | |
| Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling | [Link](https://arxiv.org/abs/2412.05271) | | | |
| VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding | [Link](https://arxiv.org/abs/2412.03735) | | | |
| VideoICL: Confidence-based Iterative In-context Learning for Out-of-Distribution Video Understanding | [Link](https://arxiv.org/abs/2412.02186) | | | |
| ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos | [Link](https://arxiv.org/abs/2411.14901) | | | |
| VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection | [Link](https://arxiv.org/abs/2411.14794) | | | |
| Adaptive Video Understanding Agent: Enhancing efficiency with dynamic frame sampling and feedback-driven reasoning | [Link](https://arxiv.org/abs/2410.20252) | | | |
| VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning with LLMs | [Link](https://arxiv.org/abs/2409.20365) | | | |
| MECD: Unlocking Multi-Event Causal Discovery in Video Reasoning | [Link](https://arxiv.org/abs/2409.17647) | | | |
| AMEGO: Active Memory from long EGOcentric videos | [Link](https://arxiv.org/abs/2409.10917) | | | |
| Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition | [Link](https://arxiv.org/abs/2501.03230) | | | |


### Benchmarks for Video Reasoning

| Title | Paper | Code | Dataset | Venue |
| :--- | :---: | :---: | :---: | :---: |
| Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding | [Link](https://arxiv.org/abs/2507.15028) | | | |
| HV-MMBench: Benchmarking MLLMs for Human-Centric Video Understanding | [Link](https://arxiv.org/abs/2507.04909) | | | |
| ImplicitQA: Going beyond frames towards Implicit Video Reasoning | [Link](https://arxiv.org/abs/2506.21742) | | | |
| Looking Beyond Visible Cues: Implicit Video Question Answering via Dual-Clue Reasoning | [Link](https://arxiv.org/abs/2506.07811) | | | |
| MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning | [Link](https://arxiv.org/abs/2506.05523) | | | |
| Go Beyond Earth: Understanding Human Actions and Scenes in Microgravity Environments | [Link](https://arxiv.org/abs/2506.02845) | | | |
| Time Blindness: Why Video-Language Models Can't See What Humans Can? | [Link](https://arxiv.org/abs/2505.24867) | | | |
| ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding | [Link](https://arxiv.org/abs/2505.23922) | | | |
| VidText: Towards Comprehensive Evaluation for Video Text Understanding | [Link](https://arxiv.org/abs/2505.22810) | | | |
| Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning? | [Link](https://arxiv.org/abs/2505.21374) | | | |
| VideoEval-Pro: Robust and Realistic Long Video Understanding Evaluation | [Link](https://arxiv.org/abs/2505.14640) | | | |
| Breaking Down Video LLM Benchmarks: Knowledge, Spatial Perception, or True Temporal Understanding? | [Link](https://arxiv.org/abs/2505.14321) | | | |
| MINERVA: Evaluating Complex Video Reasoning | [Link](https://arxiv.org/abs/2505.00681) | | | |
| IV-Bench: A Benchmark for Image-Grounded Video Perception and Reasoning in Multimodal LLMs | [Link](https://arxiv.org/abs/2504.15415) | | | |
| VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning | [Link](https://arxiv.org/abs/2504.07956) | | | |
| InstructionBench: An Instructional Video Understanding Benchmark | [Link](https://arxiv.org/abs/2504.05040) | | | |
| Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 | [Link](https://arxiv.org/abs/2503.24376) | | | |
| H2VU-Benchmark: A Comprehensive Benchmark for Hierarchical Holistic Video Understanding | [Link](https://arxiv.org/abs/2503.24008) | | | |
| OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts | [Link](https://arxiv.org/abs/2503.22952) | | | |
| V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning | [Link](https://arxiv.org/abs/2503.11495) | | | |
| Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation | [Link](https://arxiv.org/abs/2503.10691) | | | |
| Towards Fine-Grained Video Question Answering | [Link](https://arxiv.org/abs/2503.06820) | | | |
| SVBench: A Benchmark with Temporal Multi-Turn Dialogues for Streaming Video Understanding | [Link](https://arxiv.org/abs/2502.10810) | | | |
| MMVU: Measuring Expert-Level Multi-Discipline Video Understanding | [Link](https://arxiv.org/abs/2501.12380) | | | |
| OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? | [Link](https://arxiv.org/abs/2501.05510) | | | |
| HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding | [Link](https://arxiv.org/abs/2501.01645) | | | |
| Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces | [Link](https://arxiv.org/abs/2412.14171) | | | |
| Neptune: The Long Orbit to Benchmarking Long Video Understanding | [Link](https://arxiv.org/abs/2412.09582) | | | |
| 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark | [Link](https://arxiv.org/abs/2412.07825) | | | |
| Black Swan: Abductive and Defeasible Video Reasoning in Unpredictable Events | [Link](https://arxiv.org/abs/2412.05725) | | | |
| VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding | [Link](https://arxiv.org/abs/2412.03735) | | | |
| Motion-Grounded Video Reasoning: Understanding and Perceiving Motion at Pixel Level | [Link](https://arxiv.org/abs/2411.09921) | | | |
| TOMATO: Assessing Visual Temporal Reasoning Capabilities in Multimodal Foundation Models | [Link](https://arxiv.org/abs/2410.23266) | | | |
| TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models | [Link](https://arxiv.org/abs/2410.10818) | | | |
| One Token to Seg Them All: Language Instructed Reasoning Segmentation in Videos | [Link](https://arxiv.org/abs/2409.19603) | | | |
| Compositional Physical Reasoning of Objects and Events from Videos | [Link](https://arxiv.org/abs/2408.02687) | | | |
| ViLLa: Video Reasoning Segmentation with Large Language Model | [Link](https://arxiv.org/abs/2407.14500) | | | |


### Related Surveys

| Title | Paper | Code | Dataset | Venue |
| :--- | :---: | :---: | :---: | :---: |
| Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models | [Link](https://arxiv.org/abs/2505.04921) | | | |
| VideoLLM Benchmarks and Evaluation: A Survey | [Link](https://arxiv.org/abs/2505.03829) | | | |
| Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models | [Link](https://arxiv.org/abs/2504.21277) | | | |
| Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey | [Link](https://arxiv.org/abs/2503.12605) | | | |
| From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding | [Link](https://arxiv.org/abs/2409.18938) | | | |
| From Seconds to Hours: Reviewing MultiModal Large Language Models on Comprehensive Long Video Understanding | [Link](https://arxiv.org/abs/2409.18938) | | | |
| VideoQA in the Era of LLMs: An Empirical Study | [Link](https://arxiv.org/abs/2408.04223) | | | |
| Video Understanding with Large Language Models: A Survey | [Link](https://arxiv.org/abs/2312.17322) | | | |




## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yunlong10/Awesome-Video-LLM-Post-Training&type=Date)](https://star-history.com/#yunlong10/Awesome-Video-LLM-Post-Training&Date)


## 📝 Citation

```bibtex
@misc{tang2025videollmposttraining,
  title={Awesome Video-LLM Post-Training},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yunlong10/Awesome-Video-LLM-Post-Training}},
}
```
