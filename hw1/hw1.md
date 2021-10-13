## Overview of Implementation

### Files

All files needed to run your code are in the `HW1` folder, but there will be some blanks you need to fill.
These locations are marked with `TODO` and are found in the following files:

- [infrastructure/rl_trainer.py](wurl/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](wurl/infrastructure/utils.py)
- [policies/MLP_policy.py](wurl/policies/MLP_policy.py)
- [agents/pg_agent.py](wurl/agents/pg_agent.py)
- [policies/MLP_policy.py](wurl/policies/MLP_policy)

The script to run the experiments is found in [scripts/run_hw1.py](wurl/scripts/run_hw1.py)

### Overview

The main training loop is implemented in [infrastructure/rl_trainer.py](wurl/infrastructure/rl_trainer.py).

The policy gradient algorithm uses the following 3 steps:

1. **Sample trajectories** by generating rollouts under your current policy.
2. **Estimate returns and compute advantages.** This is executed in the `train` function of [pg_agent.py](wurl/agents/pg_agent.py)
3. **Train/Update parameters.** The computational graph for the policy and the baseline, as well as the update functions, are implemented in [policies/MLP_policy.py](wurl/policies/MLP_policy.py).

## Implementing Policy Gradients

You will be implementing two different return estimators within [pg_agent.py](wurl/agents/pg_agent.py). The first (“Case 1” within `calculate_q_vals`) uses the discounted cumulative return of the full trajectory and corresponds to the “vanilla” form of the policy gradient:
$$
r\left(\tau_{i}\right)=\sum_{t^{\prime}=0}^{T-1} \gamma^{t^{\prime}} r\left(s_{i t^{\prime}}, a_{i t^{\prime}}\right)
$$
The second (“Case 2”) uses the “reward-to-go” formulation:
$$
r\left(\tau_{i}\right)=\sum_{t^{\prime}=t}^{T-1} \gamma^{t^{\prime}-t} r\left(s_{i t^{\prime}}, a_{i t^{\prime}}\right)
$$
Note that these differ only by the starting point of the summation.

Implement these return estimators as well as the remaining sections marked `TODO` in the code. 
You may skip those sections that are run only if `nn_baseline` is True;
(These sections are in `MLPPolicyPG:update` and `PGAgent:estimate_advantage`.)

## Experiments

**Experiment 1 (CartPole)**. Run multiple experiments with the PG algorithm on the discrete `CartPole-v0` environment, using the following commands:

```
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-dsa --exp_name q1_sb_no_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg -dsa --exp_name q1_sb_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg --exp_name q1_sb_rtg_na

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-dsa --exp_name q1_lb_no_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-rtg -dsa --exp_name q1_lb_rtg_dsa

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
-rtg --exp_name q1_lb_rtg_na
```

What’s happening here:

- **-n** : Number of iterations.
- **-b** : Batch size (number of state-action pairs sampled while acting according to the current policy at each iteration).
- **-dsa** : Flag: if present, sets `standardize_advantages` to False. Otherwise, by default, standardizes advantages to have a mean of zero and standard deviation of one.
- **-rtg** : Flag: if present, sets `reward_to_go=True`. Otherwise, `reward_to_go=False` by default.
- **--exp_name** : Name for experiment, which goes into the name for the data logging directory.

Various other command line arguments will allow you to set batch size, learning rate, network architecture,
and more. You can change these as well, but keep them fixed between the 6 experiments mentioned above.

**Deliverables for report:**

- Create two graphs:
  - In the first graph, compare the learning curves (average return at each iteration) for the experiments prefixed with `q1_sb_`. (The small batch experiments.)
  - In the second graph, compare the learning curves for the experiments prefixed with `q1_lb_`. (The large batch experiments.)
- Answer the following questions briefly:
  - Which value estimator has better performance without advantage-standardization: the trajectory-centric one, or the one using reward-to-go?
  - Did advantage standardization help?
  - Did the batch size make an impact?
- Provide the exact command line configurations you used to run your experiments, including any parameters changed from their defaults.

**What to Expect:**

- The best configuration of `CartPole` in both the large and small batch cases should converge to a maximum score of 200.

**Experiment 2 (InvertedPendulum).** Run experiments on the `InvertedPendulum-v2` continuous control environment as follows:

```
python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b <b*> -lr <r*> -rtg \
--exp_name q2_b<b*>_r<r*>
```

where your task is to find the smallest batch size **b*** and largest learning rate **r*** that gets to optimum (maximum score of 1000) in less than 100 iterations.
The policy performance may fluctuate around 1000; this is fine.
The precision of **b*** and **r*** need only be one significant digit.

**Deliverables:**

- Given the **b*** and **r*** you found, provide a learning curve where the policy gets to optimum (maximum score of 1000) in less than 100 iterations. (This may be for a single random seed, or averaged over multiple.)
- Provide the exact command line configurations you used to run your experiments.

## Submission

### Submitting the PDF

Your report should be a document containing

1. All graphs and answers to short explanation questions requested for Experiments 1-4.
2. All command-line expressions you used to run your experiments.

### Submitting the code and experiment runs

In order to turn in your code and experiment logs, create a folder that contains the following:

- A folder named run logs with all the experiment runs from this assignment. These folders can be copied directly from the `wurl/data` folder. **Do not change the names originally assigned to the folders, as specified by expnamein the instructions.** Video logging is disabled by default in the code, but if you turned it on for debugging, you need to run those again with `--video log freq -1`, or else the file size will be too large for submission.
- The wurl folder with all the `.py` files, with the same names and directory structure as the original homework repository (excluding the `wurl/data` folder). Also include any special instructions we need to run in order to produce each of your figures or tables in the form of a `README` file.

As an example, the unzipped version of your submission should result in the following file structure.
**Make sure that the submit.zip file is below 15MB.**

```
    submit.zip
    +---run_logs
    |   +---q1_lb_rtg_na_CartPole-v0_12-09-2021_17-53-4
    |   |   +---events.out.tfevents.1567529456.e3a096ac8ff4
    |   +---q3_b40000_r0.005_LunarLanderContinuous-v2_12-09-2021_00-17-58
    |   |   +---events.out.tfevents.1567529456.e3a096ac8ff4
    |   +---...
    +---wurl
    |   +---agents
    |   |   +---bc_agent.py
    |   |   +---...
    |   +---policies
    |   |   +---...
    |   +---...
    +---README.md
    +---...
```
