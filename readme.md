# CPU-SCHEDULING

Training Deep-RL-Model to optimize the turnaround time for cpu scheduling algorithm outperforming Round-Robin.
the model learns to prioritize tasks effectively reducing the turnaround time


## SYSTEM DESIGN

```mermaid
graph TD
    A[Initialize Environment and Model] --> B[Rollout Phase]
    B --> C[Policy Update Phase]
    C --> D[Log Metrics to WandB]
    D --> E[Check if Training Complete]
    E -->|No| B
    E -->|Yes| F[Save Model Weights]

    subgraph Rollout Phase
        B1[Collect Observations] --> B2[Get Actions from Actor]
        B2 --> B3[Execute Actions in Environment]
        B3 --> B4[Collect Rewards]
        B4 --> B5[Compute Rewards-to-Go]
    end

    subgraph Policy Update Phase
        C1[Evaluate Value Function and Log Probs] --> C2[Compute Advantage]
        C2 --> C3[Compute Surrogate Losses]
        C3 --> C4[Update Actor and Critic Networks]
    end

    subgraph Get Actions from Actor
        B21[Pass Observations through Actor Network] --> B22[Create Multivariate Normal Distribution]
        B22 --> B23[Sample Action from Distribution]
        B23 --> B24[Compute Log Probability of Action]
    end

    subgraph Execute Actions in Environment
        B31[Step Environment with Action] --> B32[Get New Observation and Reward]
        B32 --> B33[Update Environment State]
    end

    subgraph Update Actor and Critic Networks
        C41[Zero Gradients] --> C42[Backward Pass]
        C42 --> C43[Step Optimizer]
    end

    A -->|Initialize| G[FeedForwardNN Class]
    A -->|Initialize| H[PPO Class]
    A -->|Initialize| I[PrioritySchedulerEnv Class]

    G -->|Actor Network| B21
    G -->|Critic Network| C1

    H -->|Actor Optimizer| C4
    H -->|Critic Optimizer| C4

    I -->|Environment Interaction| B31
    I -->|Environment Interaction| B32
    I -->|Environment Interaction| B33
```