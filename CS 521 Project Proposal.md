Can keep some notes and stuff here for now, eventually write the proposal:

RL Based Cost Optimization:  
Paper: [https://mlforsystems.org/assets/papers/neurips2022/paper11.pdf\#:\~:text=level%20phases,deep%20RL%20algorithms%20to%20further](https://mlforsystems.org/assets/papers/neurips2022/paper11.pdf#:~:text=level%20phases,deep%20RL%20algorithms%20to%20further)

From Texts:

- XLA Optimization using reinforcement learning, could try to reimplement this or use it on a quant specific algorithm  
- I think what they do in this paper (use an RL algorithm to come up with an XLA cost model) could potentially be feasible and we could pick some various workloads to do (AI based, Quant based, etc.). Relates to the first project example in the example projects part of the assignment description  
- The RL thing seems pretty interesting cause it seems like it's got a huge amount of potential  
- Reminds me of AlphaTensor which did something similar iirc, used RL to find a state of the art matrix multiplication optimization for small tensors  
- The first seems pretty cool could definitely apply to hft for volatile stocks  
- It’s really computationally expensive to make trading strategies for that case so there’s room for energy and memory optimizations 

HFT / quant related workloads might be tough to maintain high accuracy/yields while optimizing the underlying operations. Based on the timeline for the project it might be better to switch to a AI powered workload, high level synthesis seems to be easily optimized at the compiler level. Optimizations for HLS have been shown to translate to better hardware performance (energy consumption, memory overhead, runtime metrics). We can do other AI workloads or a quant focused one, but with quant we would need to show metrics about the compiler performance of the workload while still maintaining the algos ability to create strong predictions and generate a profit.

HLS Workload Papers  
[https://research.redhat.com/blog/publication/reinforcement-learning-strategies-for-compiler-optimization-in-high-level-synthesis/](https://research.redhat.com/blog/publication/reinforcement-learning-strategies-for-compiler-optimization-in-high-level-synthesis/?utm_source=chatgpt.com)

[https://arxiv.org/abs/1901.04615](https://arxiv.org/abs/1901.04615?utm_source=chatgpt.com)

High-Level Synthesis (HLS) is an automated design process that translates behavioral descriptions of digital systems, typically written in high-level programming languages like C, C++, or SystemC, into register-transfer level (RTL) representations suitable for hardware implementation on platforms such as FPGAs (Field-Programmable Gate Arrays) and ASICs (Application-Specific Integrated Circuits).

**Team Members:** Erik Wilson, Ahyush Kaul, Rayaan Farqui

**Project Proposal: RL-based Pass Ordering Optimization for XLA**

**Problem Description:**  
Modern machine learning compilers like XLA rely on a sequence of optimization passes to transform computation graphs into efficient executable code. However, the default, heuristic-based pass ordering can be suboptimal and result in increased operation counts and slower execution. Our project addresses this problem by developing an offline reinforcement learning framework to learn and predict an optimal sequence of HLO passes that reduce computation cost metrics, such as operation count, across diverse model graphs.

**Summary of Implementation:**  
We propose to build an offline RL pipeline trained on features extracted from a collection of HLO graphs generated from selected math-intensive workloads based on Ganai, ‘23. The agent will learn a mapping from graph features to a sequence of optimization passes that yield improved performance. After training, the optimal pass sequences will be extracted into a lookup or configuration mapping. We will then simulate applying these sequences on a held-out test set of HLO graphs to measure improvements relative to the default pass ordering, expanding on the current workflow by experimenting to find an optimal reward function structure.

**Work Breakdown:**

* Identify ML workloads to study within our reinforcement learning optimizations.  
* Create a simplified environment to extract key HLO graph features.  
* Define the state space (HLO features), action space (available HLO passes), and a reward function based on cost reduction.  
* Implement an RL training loop to train an agent on the environment.  
* Derive a mapping from graph to optimal pass sequences using the trained policy.  
* Apply the learned sequences to test HLO graphs and evaluate improvements.

References:  
Ganai, M., Li, H., Enns, T., Wang, Y., & Huang, R. (2023). *Target-independent XLA optimization using Reinforcement Learning.* arXiv preprint arXiv:2308.14364. Retrieved from [https://arxiv.org/abs/2308.14364](https://arxiv.org/abs/2308.14364)