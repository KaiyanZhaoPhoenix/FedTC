import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import swanlab as wandb

from src.clients.clientfedtc import clientFedTC
from src.servers.serverbase import Server

# --- RL Components (DDPG) ---

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)
        self.max_action = max_action

        # [初始化优化] 确保初始动作 (lambda, tau) 很小 (约 0.1~0.15)
        # 防止热身结束后 RL 立即进行激进过滤
        nn.init.uniform_(self.l3.weight, -0.003, 0.003)
        nn.init.constant_(self.l3.bias, -2.0) 

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        # 输出范围映射到 (0, 1) * max_action
        return torch.sigmoid(self.l3(a)) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = []
        self.batch_size = 16
        self.discount = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        state, action, reward, next_state = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)

        # Critic Update
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (self.discount * target_Q).detach()
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target Soft Update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# --- FedTC Server Logic ---

class FedTC(Server):
    def __init__(self, args, times) -> None:
        super(FedTC, self).__init__(args, times)
        self.set_clients(clientFedTC)

        # RL Config
        # State: [mean(delta), std(delta), mean(util), std(util), history] = 5 dims
        # Action: [lambda, tau] = 2 dims
        self.rl_agent = DDPGAgent(state_dim=5, action_dim=2, device=self.device)
        self.prev_state = None
        self.prev_action = None

        # Initialize Global Momentum
        self.global_momentum = {} 
        for k, v in self.global_model.state_dict().items():
            self.global_momentum[k] = torch.zeros_like(v).to(self.device)

        self.history_metric = 0.0
        # FedTC hyperparameters (configurable)
        self.beta = getattr(args, 'fedtc_beta', None)
        if self.beta is None:
            self.beta = getattr(args, 'beta', 0.1)

        self.warm_up_rounds = getattr(args, 'fedtc_warmup_rounds', 10)
        self.noise_init = getattr(args, 'fedtc_noise_init', 0.05)
        self.noise_decay = getattr(args, 'fedtc_noise_decay', 0.98)
        self.noise_min = getattr(args, 'fedtc_noise_min', 0.005)
        self.tau_max = getattr(args, 'fedtc_tau_max', 1.0)
        self.server_update_lr = getattr(args, 'fedtc_server_update_lr', 1.0)
        self.gamma_fairness = getattr(args, 'fedtc_gamma_fairness', 0.0)
        self.gamma_entropy = getattr(args, 'fedtc_gamma_entropy', 0.0)

    def fit(self):
        test_acc = self.test_metrics()
        self.logger.info(f"Initial Global Model Test Accuracy: {test_acc}")
        wandb.log({"round": 0, "global_model_test_acc": test_acc})

        # [策略 1] Warm-up: 前 N 轮不进行过滤
        warm_up_rounds = self.warm_up_rounds 

        for cr in range(self.communication_rounds):
            self.logger.info(f"\n-------------Round number: {cr+1}-------------")
            s_t = time.time()

            # 1. Client Sampling & Training
            self.send_models()
            self.select_clients()

            updates_deltas = [] 
            utilities = []      

            self.logger.info("Clients training...")
            for idx in self.selected_clients:
                self.clients[idx].train()
                updates_deltas.append(self.clients[idx].model_delta)
                utilities.append(self.clients[idx].utility)

            # 2. Evidence Extraction (Robust Centroid & Deviations)
            layer_keys = updates_deltas[0].keys()
            robust_centroid = {}

            # Using Coordinate-wise Median as efficient Robust Centroid approximation
            for k in layer_keys:
                layer_stack = torch.stack([d[k].to(self.device) for d in updates_deltas])
                robust_centroid[k] = torch.median(layer_stack, dim=0).values

            deviations = []
            for i in range(len(updates_deltas)):
                dist_sq = 0.0
                for k in layer_keys:
                    diff = updates_deltas[i][k].to(self.device) - robust_centroid[k]
                    dist_sq += torch.sum(diff ** 2).item()
                deviations.append(np.sqrt(dist_sq))
            deviations = np.array(deviations)

            # [策略 2] 数值稳定性缩放: 让 deviations 分布在 tanh 敏感区间
            avg_dev = np.mean(deviations) + 1e-6
            scaled_deviations = deviations / avg_dev

            # Normalize Utilities
            u_min, u_max = min(utilities), max(utilities)
            if u_max - u_min < 1e-9:
                norm_utilities = np.zeros_like(utilities)
            else:
                norm_utilities = np.array([(u - u_min) / (u_max - u_min) for u in utilities])

            # 3. RL State Construction
            current_state = np.array([
                np.mean(scaled_deviations), np.std(scaled_deviations),
                np.mean(norm_utilities), np.std(norm_utilities),
                self.history_metric
            ])

            # 4. RL Action Selection (with Warm-up and Decaying Noise)
            if cr < warm_up_rounds:
                # [Warm-up Phase]
                lambda_t = 0.0
                tau_t = 0.0
                action = np.array([0.0, 0.0]) # Pseudo action for buffer
                self.logger.info(f"Warm-up Phase ({cr+1}/{warm_up_rounds}): Force lambda=0, tau=0")
            else:
                # [Normal Phase]
                raw_action = self.rl_agent.select_action(current_state)

                # [策略 3] 噪声衰减 (Noise Decay)
                # 可配置噪声，指数衰减到下限
                current_noise_std = max(self.noise_min, self.noise_init * (self.noise_decay ** (cr - warm_up_rounds)))
                noise = np.random.normal(0, current_noise_std, size=raw_action.shape)

                action = raw_action + noise
                lambda_t = np.clip(action[0], 0.0, 1.0)
                tau_t = np.clip(action[1], 0.0, self.tau_max)

                self.logger.info(f"RL Action (noise_std={current_noise_std:.4f}): lambda={lambda_t:.3f}, tau={tau_t:.3f}")

            # 5. Trust Calibration (Credibility Scores)
            credibilities = []
            for i in range(len(self.selected_clients)):
                score = (1 - lambda_t) * norm_utilities[i] - lambda_t * np.tanh(scaled_deviations[i]) - tau_t
                credibilities.append(max(0, score))

            sum_cred = sum(credibilities)
            if sum_cred > 1e-9:
                alpha_weights = [c / sum_cred for c in credibilities]
            else:
                self.logger.warning("All clients filtered! Using uniform weights.")
                alpha_weights = [1.0 / len(credibilities) for _ in credibilities]

            self.logger.info(f"Credibilities: {[f'{c:.3f}' for c in credibilities]}")

            accepted_indices = [i for i, c in enumerate(credibilities) if c > 0]
            accepted_ratio = len(accepted_indices) / max(1, len(credibilities))
            nonzero_weights = [alpha_weights[i] for i in accepted_indices] if len(accepted_indices) > 0 else []
            if len(nonzero_weights) > 1:
                w = np.array(nonzero_weights, dtype=np.float64)
                w_entropy = -np.sum(w * np.log(w + 1e-12))
                w_entropy_norm = w_entropy / np.log(len(nonzero_weights) + 1e-12)
            else:
                w_entropy_norm = 0.0

            # 6. Weighted Aggregation
            global_delta = {}
            for k in layer_keys:
                global_delta[k] = torch.zeros_like(updates_deltas[0][k]).to(self.device)
                for i, w in enumerate(alpha_weights):
                    if w > 0:
                        global_delta[k] += w * updates_deltas[i][k].to(self.device)

            # Apply update
            for name, param in self.global_model.named_parameters():
                if name in global_delta:
                    param.data += self.server_update_lr * global_delta[name]

            # 7. Reward Calculation (Targeting lower Beta for stability)
            # Cosine Alignment
            flat_delta = torch.cat([global_delta[k].view(-1) for k in layer_keys])
            flat_momentum = torch.cat([self.global_momentum[k].view(-1) for k in layer_keys])

            if torch.norm(flat_momentum) > 1e-9 and torch.norm(flat_delta) > 1e-9:
                cos_sim = torch.cosine_similarity(flat_delta.unsqueeze(0), flat_momentum.unsqueeze(0)).item()
            else:
                cos_sim = 0.0

            # Dispersion
            if len(accepted_indices) > 0:
                disp_sq_sum = 0.0
                for idx in accepted_indices:
                    d_sq = 0.0
                    for k in layer_keys:
                         diff = updates_deltas[idx][k].to(self.device) - global_delta[k]
                         d_sq += torch.sum(diff**2).item()
                    disp_sq_sum += d_sq
                dispersion = np.sqrt(disp_sq_sum / len(accepted_indices))
            else:
                dispersion = 0.0

            # Reward = Alignment - beta * Dispersion
            if len(accepted_indices) > 0:
                norm_utils_accepted = np.array([norm_utilities[i] for i in accepted_indices])
                fairness_score = max(0.0, 1.0 - float(np.std(norm_utils_accepted)))
            else:
                fairness_score = 0.0

            reward = cos_sim - self.beta * dispersion + self.gamma_fairness * fairness_score + self.gamma_entropy * w_entropy_norm

            # RL Training Step
            if self.prev_state is not None:
                self.rl_agent.replay_buffer.append((self.prev_state, self.prev_action, reward, current_state))
                self.rl_agent.train()

            self.prev_state = current_state
            self.prev_action = action

            # Update History and Momentum
            for k in layer_keys:
                self.global_momentum[k] = 0.9 * self.global_momentum[k] + global_delta[k]

            self.history_metric = 0.9 * self.history_metric + 0.1 * cos_sim

            # Logging
            self.budget.append(time.time() - s_t)
            test_acc = self.test_metrics()
            self.logger.info(f"Global Test Acc: {test_acc:.4f}, Reward: {reward:.4f}")
            wandb.log({
                "round": cr + 1,
                "reward": reward,
                "lambda": lambda_t,
                "tau": tau_t,
                "accepted_ratio": accepted_ratio,
                "weight_entropy": w_entropy_norm,
                "fairness_score": fairness_score,
                "global_model_test_acc": test_acc,
            })