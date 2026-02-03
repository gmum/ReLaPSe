<div align="center"> 

# ReLaPSe: Reinforcement-Learning-trained Adversarial Prompt Search for Erased concepts in unlearned diffusion models 

</div>

<p align="center">
  <a href="https://www.arxiv.org/abs/2602.00350"><img src="https://img.shields.io/badge/arXiv-2602.00350-b31b1b.svg" alt="arXiv"></a>
</p>


**Abstract:** Machine unlearning is a key defense mechanism for removing unauthorized concepts from text-to-image diffusion models, yet recent evidence shows that latent visual information often persists after unlearning. Existing adversarial approaches for exploiting this leakage are constrained by fundamental limitations: optimization-based methods are computationally expensive due to per-instance iterative search. At the same time, reasoning-based and heuristic techniques lack direct feedback from the target model’s latent visual representations. To address these challenges, we introduce ReLaPSe, a policy-based adversarial framework that reformulates concept restoration as a reinforcement learning problem. ReLaPSe trains an agent using Reinforcement Learning with Verifiable Rewards (RLVR), leveraging the diffusion model’s noise prediction loss as a model-intrinsic and verifiable feedback signal. This closed-loop design directly aligns textual prompt manipulation with latent visual residuals, enabling the agent to learn transferable restoration strategies rather than optimizing isolated prompts. By pioneering the shift from per-instance optimization to global policy learning, ReLaPSe achieves efficient, near-real-time recovery of fine-grained identities and styles across multiple state-of-the-art unlearning methods, providing a scalable tool for rigorous red-teaming of unlearned diffusion models.Some experimental evaluations involve sensitive visual concepts, such as nudity.


## 🚧 Code coming soon
This repository will be updated with the full implementation shortly.
