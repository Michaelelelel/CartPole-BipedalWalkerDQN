## Evaluation Results and Discussion

### Evaluation Setup

After training, selected DQN configurations were evaluated with a greedy policy. Exploration was disabled during evaluation in order to measure the quality of the learned policy itself rather than exploratory training behavior.

CartPole-v1 configurations were evaluated for 100 episodes. BipedalWalker-v3 configurations were evaluated for 50 episodes. The main evaluation metric is the cumulative episode return, while episode length is used as an additional supporting metric. For each run, the mean, standard deviation, median, minimum, maximum, and best observed return were recorded.

This setup allows a fair comparison within each environment because models from the same environment are assessed with the same deterministic action-selection rule.

---

### CartPole-v1 Sanity Check

CartPole-v1 is the simpler benchmark environment in this project. The best CartPole configurations reached the maximum return of `500.00` over all 100 greedy evaluation episodes.

| Configuration | Mean Return | Std. Return | Median Return | Best Return | Mean Length |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cp_arch_large_seed42` | **500.00** | 0.00 | 500.00 | 500.00 | 500.00 |
| `cp_policy_epsilon_greedy_seed42` | **500.00** | 0.00 | 500.00 | 500.00 | 500.00 |
| `cp_explore_conservative_seed42` | 493.84 | 24.02 | 500.00 | 500.00 | 493.84 |

These results show that the implementation can solve CartPole-v1 reliably and that the remaining analysis can focus on the more difficult BipedalWalker-v3 extension.

---

### BipedalWalker-v3 Overall Ranking

The BipedalWalker-v3 evaluation clearly shows that the best-performing configuration is `bw_boltzmann_final_01_seed42`. It achieves a mean return of `329.16 ± 49.99` over 50 greedy evaluation episodes, which is substantially higher than the other tested configurations.

| Rank | Configuration | Policy | Mean Return | Std. Return | Median Return | Best Return | Mean Length |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `bw_boltzmann_final_01_seed42` | Boltzmann | **329.16** | 49.99 | 321.32 | 513.25 | 143.38 |
| 2 | `bw_final_02_balanced_replay_explore_400_seed42` | Epsilon-greedy | 242.98 | 50.84 | 225.43 | 413.00 | 132.04 |
| 3 | `bw_03_more_exploration_large_300` | Epsilon-greedy | 163.71 | 12.57 | 164.61 | 189.30 | 89.52 |
| 4 | `bw_boltzmann_04_more_exploration_temp010_300_seed24` | Boltzmann | 160.57 | 14.96 | 159.09 | 195.71 | 88.98 |
| 5 | `bw_01_baseline_large_300` | Epsilon-greedy | 158.65 | 16.78 | 154.96 | 203.43 | 89.10 |

The final Boltzmann configuration outperforms the best epsilon-greedy configuration by `+86.18` mean return, corresponding to an improvement of approximately `+35%`.

---

### Main Findings

#### 1. The final Boltzmann configuration is the clear best model

The strongest result is achieved by `bw_boltzmann_final_01_seed42`. Its mean return of `329.16` is higher than the best epsilon-greedy configuration and nearly twice as high as the original epsilon-greedy baseline. The median return of `321.32` confirms that this result is not only caused by a few unusually good episodes, but that the model generally performs at a much higher level.

The best single evaluation episode reached a return of `513.25`, which indicates that the learned policy is capable of producing highly effective walking behavior in the discretized BipedalWalker environment.

This configuration uses:

| Parameter | Value |
| --- | ---: |
| Policy | Boltzmann |
| Network | `mlp_large` |
| Learning rate | `0.00005` |
| Gamma | `0.99` |
| Replay buffer size | `500000` |
| Batch size | `256` |
| Warmup steps | `50000` |
| Total epochs | `500` |
| Updates per epoch | `500` |
| Exploration start | `2.0` |
| Exploration end | `0.2` |
| Seed | `42` |

The combination of a low learning rate, large replay buffer, extended warmup phase, large batch size, longer training budget, and Boltzmann exploration appears to have produced the most stable and effective training outcome.

---

#### 2. Boltzmann exploration became highly effective after tuning

Earlier Boltzmann configurations performed similarly to or slightly below the epsilon-greedy baseline. For example, `bw_06_boltzmann_large_300` achieved a mean return of `145.10`, which is below the epsilon-greedy baseline result of `158.65`.

However, the tuned final Boltzmann configuration reached `329.16`, showing that Boltzmann exploration can be highly effective when combined with suitable training settings.

| Configuration | Mean Return | Std. Return |
| --- | ---: | ---: |
| `bw_06_boltzmann_large_300` | 145.10 | 12.91 |
| `bw_boltzmann_01_temp15_temp015_250_seed22` | 117.09 | 19.69 |
| `bw_boltzmann_02_gamma995_temp015_250_seed21` | 130.45 | 30.50 |
| `bw_boltzmann_03_fast_lr_temp015_300_seed23` | 146.62 | 7.27 |
| `bw_boltzmann_04_more_exploration_temp010_300_seed24` | 160.57 | 14.96 |
| `bw_boltzmann_final_01_seed42` | **329.16** | 49.99 |

This suggests that the exploration strategy alone is not sufficient. Final performance depends strongly on the interaction between exploration schedule, learning rate, replay buffer size, warmup duration, and training budget.

---

#### 3. The best epsilon-greedy configuration improved strongly over the baseline

Among the epsilon-greedy configurations, `bw_final_02_balanced_replay_explore_400_seed42` achieved the best result with a mean return of `242.98 ± 50.84`.

Compared to the baseline `bw_01_baseline_large_300`, which achieved `158.65 ± 16.78`, this is a large improvement in average return. The higher standard deviation also shows that the final epsilon-greedy configuration is less consistent than the baseline, but it is clearly capable of producing much stronger episodes.

| Configuration | Mean Return | Std. Return | Median Return | Best Return |
| --- | ---: | ---: | ---: | ---: |
| `bw_01_baseline_large_300` | 158.65 | 16.78 | 154.96 | 203.43 |
| `bw_final_02_balanced_replay_explore_400_seed42` | **242.98** | 50.84 | 225.43 | 413.00 |

This shows that epsilon-greedy can learn substantially better behavior after stronger tuning, even though it did not reach the final Boltzmann configuration.

---

#### 4. More exploration improved the original epsilon-greedy baseline

The configuration `bw_03_more_exploration_large_300` improved over the baseline by increasing exploration. It achieved a mean return of `163.71`, compared to the baseline score of `158.65`.

| Configuration | Explore End | Mean Return |
| --- | ---: | ---: |
| `bw_01_baseline_large_300` | 0.05 | 158.65 |
| `bw_03_more_exploration_large_300` | 0.10 | **163.71** |

This suggests that slightly stronger exploration during training was beneficial. The agent likely discovered more diverse transitions and avoided premature convergence to weaker behavior.

However, the improvement is moderate, meaning that exploration alone was not enough to produce a major performance jump.

---

#### 5. Lowering the learning rate alone did not improve the baseline

The configuration `bw_02_stable_low_lr_large_300` reduced the learning rate from `0.0003` to `0.0001`. This resulted in a lower mean return of `141.35`, compared to the baseline score of `158.65`.

| Configuration | Learning Rate | Mean Return |
| --- | ---: | ---: |
| `bw_01_baseline_large_300` | 0.00030 | **158.65** |
| `bw_02_stable_low_lr_large_300` | 0.00010 | 141.35 |

This indicates that simply reducing the learning rate made learning more conservative, but did not improve final performance under the same training budget.

Interestingly, the best overall configuration uses an even lower learning rate of `0.00005`, but also changes several other factors such as buffer size, warmup duration, batch size, and training length. Therefore, the success of the final Boltzmann run cannot be attributed to the learning rate alone.

---

#### 6. The medium-sized network performed worse than the large network

The network-size comparison shows that the large MLP performed better than the medium MLP.

| Configuration | Network | Mean Return |
| --- | --- | ---: |
| `bw_01_baseline_large_300` | `mlp_large` | **158.65** |
| `bw_04_medium_network_300` | `mlp_medium` | 132.81 |

The lower performance of the medium network suggests that the task benefits from a larger function approximator. Since BipedalWalker has more complex dynamics than simpler environments such as CartPole, the larger network likely provides better representational capacity for approximating the Q-function.

---

#### 7. Reducing gamma to 0.97 strongly hurt performance

The weakest main ablation is `bw_07_gamma_097_ablation_300`, which achieved only `93.41 ± 33.83`.

| Configuration | Gamma | Mean Return |
| --- | ---: | ---: |
| `bw_01_baseline_large_300` | 0.99 | **158.65** |
| `bw_07_gamma_097_ablation_300` | 0.97 | 93.41 |

This suggests that a high discount factor is important for this task. Since walking requires long-term coordination and delayed benefits, reducing gamma likely made the agent too short-sighted.

The high standard deviation of this run also indicates unstable evaluation behavior.

---

#### 8. Larger replay and batch size helped only moderately in the epsilon-greedy setting

The configuration `bw_05_big_replay_batch_300` increased the replay buffer size and batch size compared to the baseline. It achieved a mean return of `154.32`, which is slightly below the baseline result of `158.65`.

| Configuration | Buffer Size | Batch Size | Mean Return |
| --- | ---: | ---: | ---: |
| `bw_01_baseline_large_300` | 100000 | 128 | **158.65** |
| `bw_05_big_replay_batch_300` | 200000 | 256 | 154.32 |

This suggests that increasing replay capacity and batch size alone did not significantly improve performance in the epsilon-greedy setup. However, these components may still be important when combined with other changes, as seen in the final Boltzmann configuration and the best tuned epsilon-greedy run.

---

### Interpretation of Episode Length

Episode length generally increased with better returns, but it should not be interpreted as the main performance metric. For BipedalWalker, an agent may survive longer without necessarily achieving strong forward movement or efficient behavior. Therefore, cumulative return is the more meaningful primary metric.

The strongest runs have both higher returns and longer episodes:

| Configuration | Mean Return | Mean Length |
| --- | ---: | ---: |
| `bw_boltzmann_final_01_seed42` | **329.16** | **143.38** |
| `bw_final_02_balanced_replay_explore_400_seed42` | 242.98 | 132.04 |
| `bw_03_more_exploration_large_300` | 163.71 | 89.52 |

This indicates that the best models not only survived longer, but also achieved substantially better cumulative reward.

---

### Limitations

Although the evaluation results are strong, especially for the final Boltzmann run, some limitations should be considered.

First, not every comparison is a perfectly isolated ablation. Some configurations differ in multiple hyperparameters at once, such as learning rate, buffer size, batch size, warmup duration, number of epochs, and seed. Therefore, the results should be interpreted as configuration-level comparisons rather than strictly causal single-parameter studies.

Second, several runs use different random seeds. Since reinforcement learning can be sensitive to initialization and environment randomness, a more rigorous analysis would repeat the best configurations across multiple seeds and report the mean and standard deviation across seeds.

Third, the final Boltzmann configuration is much stronger than most others, but it also uses a larger training setup. Its superior performance is therefore likely caused by a combination of better exploration, more stable learning, a larger replay buffer, longer warmup, and a larger training budget.

---

### Conclusion

The evaluation results show that the DQN implementation successfully solves CartPole-v1 and learns meaningful control behavior in the discretized BipedalWalker-v3 environment.

The results indicate that Boltzmann exploration can be highly effective when combined with a sufficiently large replay buffer, low learning rate, extended warmup phase, large batch size, and longer training. The tuned epsilon-greedy configuration also improved substantially over the baseline, while individual ablations such as lowering gamma, using a smaller network, or changing only the learning rate did not improve performance.

Overall, the experiments provide a clear comparison of multiple DQN configurations and demonstrate the importance of careful hyperparameter tuning for reinforcement learning performance.
