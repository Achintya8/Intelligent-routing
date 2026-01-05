# AI Agents - Line-by-Line Explanation

## File 1: `rl_agent.py` - Basic Q-Learning Agent

### Lines 1-2: Imports
```python
import random
import numpy as np
```
- `random`: For epsilon-greedy action selection
- `numpy`: For numerical operations (though barely used here)

### Line 4: `class QLearningAgent:`
- Implements tabular Q-Learning algorithm
- Stores Q-values in a dictionary

### Line 5: `def __init__(self, nodes, alpha=0.1, gamma=0.9, epsilon=0.1):`
- **Parameters**:
  - `nodes`: List of all node IDs in network
  - `alpha`: Learning rate (0.1 = 10% update per experience)
  - `gamma`: Discount factor (0.9 = values future rewards at 90%)
  - `epsilon`: Exploration rate (0.1 = 10% random actions)

### Lines 6-11: Docstring
- Explains Q-Learning for routing context

### Lines 13-15: Store hyperparameters
```python
self.alpha = alpha
self.gamma = gamma
self.epsilon = epsilon
```

### Line 16: `self.nodes = nodes`
- Stores list of all network nodes

### Lines 18-20: Q-Table initialization
```python
# Q-Table: Q[current_node][next_hop] -> Value
# We can use a dict of dicts for flexibility
self.q_table = {node: {} for node in nodes}
```
- **Structure**: `{current_node: {next_hop: q_value}}`
- **Example**: `{0: {1: 0.5, 2: -0.3}, 1: {3: 0.8}}`
- Dictionary allows sparse storage (only store visited state-action pairs)

### Lines 22-29: Comments about lazy initialization
- Q-values start at 0.0 (implicit via `.get()` default)

---

### Line 31: `def get_q_value(self, state, action):`
- Retrieves Q-value for state-action pair
- **`state`**: Current node
- **`action`**: Next hop neighbor

### Line 32: `"""State = current node, Action = next hop neighbor"""`

### Line 33: `return self.q_table[state].get(action, 0.0)`
- Uses `.get()` with default 0.0
- If `action` not in inner dict, returns 0.0 (optimistic initialization)

---

### Line 35: `def choose_action(self, current_node, neighbors, target_node=None, avoid_nodes=None):`
- **Epsilon-greedy policy**
- **Parameters**:
  - `current_node`: Where packet currently is
  - `neighbors`: Valid next hops (list of node IDs)
  - `target_node`: Destination (unused in basic Q-Learning)
  - `avoid_nodes`: Set of nodes to exclude (prevents loops)

### Line 36-37: `"""Epsilon-greedy selection of next hop."""`

### Lines 39-40:
```python
if not neighbors:
    return None
```
- Dead end handling

### Lines 42-44: Filter avoided nodes
```python
valid_neighbors = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
if not valid_neighbors:
     return None
```
- List comprehension filters out visited nodes

### Line 46: `if random.random() < self.epsilon:`
- **Exploration**: With ε probability, choose randomly

### Line 47-48: `# Explore` `return random.choice(valid_neighbors)`
- Uniform random selection among valid neighbors

### Line 49: `else:`
- **Exploitation**: Choose best action by Q-values

### Line 50-51: `# Exploit: Choose neighbor with max Q-value`
```python
q_values = [self.get_q_value(current_node, n) for n in valid_neighbors]
```
- Builds list of Q-values for each valid neighbor

### Line 52: `max_q = max(q_values)`
- Finds highest Q-value

### Lines 54-56: Tie-breaking
```python
# Tie-breaking
best_actions = [n for n, q in zip(valid_neighbors, q_values) if q == max_q]
return random.choice(best_actions)
```
- If multiple neighbors have same max Q, randomly pick one
- Prevents bias from list ordering

---

### Line 58: `def learn(self, state, action, reward, next_state, next_neighbors, target_node=None):`
- **Q-Learning update rule**
- **Parameters**:
  - `state`: Node before action
  - `action`: Next hop taken
  - `reward`: Immediate reward received
  - `next_state`: Node after action
  - `next_neighbors`: Valid next hops from `next_state`
  - `target_node`: Destination (unused)

### Lines 59-62: Docstring with formula
```python
"""
Q-Learning Update Rule:
Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s,a)]
"""
```
- **Bellman optimality equation**

### Line 63: `current_q = self.get_q_value(state, action)`
- Gets current Q-value estimate

### Lines 65-67: Compute max future Q
```python
# Calculate max Q for next state
if not next_neighbors:
    max_next_q = 0.0 # Terminal or dead end
```
- If no outgoing edges, future value is 0

### Lines 68-69:
```python
else:
    max_next_q = max([self.get_q_value(next_state, n) for n in next_neighbors])
```
- Finds best Q-value achievable from next state
- **Bootstrap**: Uses current estimates

### Line 71: `new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)`
- **TD Update**:
  - `reward + gamma * max_next_q`: TD target
  - `(target - current_q)`: TD error
  - `alpha * error`: Learning step
  
### Line 72: `self.q_table[state][action] = new_q`
- Stores updated Q-value

---

## File 2: `advanced_agents.py` - Deep Learning Agents

### Lines 1-6: Imports
```python
import numpy as np
import random
import logging
from collections import deque
```
- `deque`: For experience replay buffer (fast FIFO queue)

### Line 6: `logger = logging.getLogger("AdvancedAgents")`

### Lines 8-16: PyTorch import with fallback
```python
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. Advanced agents (DQN, GNN) will act randomly or fail.")
```
- Gracefully handles missing PyTorch
- Sets flag to disable deep learning features

---

## DQN Agent (Deep Q-Network)

### Line 18: `class DQNAgent:`
- Neural network-based Q-Learning
- Replaces Q-table with function approximator

### Line 19: `def __init__(self, nodes, alpha=0.001, gamma=0.95, epsilon=1.0, hidden_dim=128, memory_size=2000, batch_size=32):`
- **New parameters**:
  - `alpha=0.001`: Lower learning rate for neural nets
  - `epsilon=1.0`: Start with full exploration, decay later
  - `hidden_dim=128`: Neural network layer size
  - `memory_size=2000`: Replay buffer capacity
  - `batch_size=32`: Mini-batch size for training

### Lines 20-23: Node indexing
```python
self.nodes = nodes
self.node_to_idx = {node: i for i, node in enumerate(nodes)}
self.idx_to_node = {i: node for i, node in enumerate(nodes)}
self.num_nodes = len(nodes)
```
- Creates bidirectional mapping (node ID ↔ index)
- Neural nets need integer indices

### Lines 25-29: Store hyperparameters and buffers
```python
self.alpha = alpha
self.gamma = gamma
self.epsilon = epsilon
self.batch_size = batch_size
self.memory = deque(maxlen=memory_size)
```
- `deque(maxlen=...)`: Automatically removes oldest when full

### Lines 31-40: Main neural network
```python
if TORCH_AVAILABLE:
    # Input: One-hot current + One-hot target (Size: 2 * num_nodes)
    # Output: Q-value for each neighbor (Size: num_nodes, masked later)
    self.model = nn.Sequential(
        nn.Linear(2 * self.num_nodes, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, self.num_nodes)
    )
```
- **Architecture**: `[2N] → [128] → [128] → [N]`
- **Input**: Concatenated one-hot vectors of current + target
- **Output**: Q-value for each possible next hop

### Lines 42-51: Target network
```python
# Target Network
self.target_model = nn.Sequential(
    nn.Linear(2 * self.num_nodes, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, self.num_nodes)
)
self.target_model.load_state_dict(self.model.state_dict())
self.target_model.eval()
```
- **Double DQN**: Separate network for computing targets
- Initialized as copy of main network
- `.eval()`: Sets to evaluation mode (disables dropout/batchnorm)

### Lines 53-54: Optimizer and loss
```python
self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
self.criterion = nn.MSELoss()
```
- Adam: Adaptive learning rate optimizer
- MSE: Mean Squared Error between predicted and target Q-values

---

### Line 56: `def _get_state_vector(self, current_node, target_node):`
- Converts (current, target) pair to neural net input

### Lines 57-62:
```python
state = torch.zeros(2 * self.num_nodes)
if current_node in self.node_to_idx:
    state[self.node_to_idx[current_node]] = 1.0
if target_node in self.node_to_idx:
    state[self.num_nodes + self.node_to_idx[target_node]] = 1.0
return state
```
- **One-hot encoding**: Vector with 1 at position corresponding to node
- **Example** (4 nodes, current=1, target=3):
  - `[0, 1, 0, 0, 0, 0, 0, 1]`
  - First half: current node
  - Second half: target node

---

### Line 64: `def choose_action(self, current_node, neighbors, target_node, avoid_nodes=None):`
- Action selection with neural network

### Lines 65-66: Handle missing neighbors

### Lines 68-72: Fallback if PyTorch unavailable
```python
if not TORCH_AVAILABLE:
    valid = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
    if not valid: return None
    return random.choice(valid)
```

### Lines 74-78: Exploration
```python
# Explore
if random.random() < self.epsilon:
    valid = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
    if not valid: return None
    return random.choice(valid)
```

### Lines 80-100: Exploitation with masking
```python
# Exploit
with torch.no_grad():
    state_vec = self._get_state_vector(current_node, target_node)
    q_values = self.model(state_vec) # [num_nodes]
    
    # Mask invalid neighbors (set to -inf)
    masked_q = torch.full_like(q_values, -float('inf'))
    found_valid = False
    for n in neighbors:
        if avoid_nodes and n in avoid_nodes:
            continue
        if n in self.node_to_idx:
            idx = self.node_to_idx[n]
            masked_q[idx] = q_values[idx]
            found_valid = True
    
    if not found_valid:
        return None

    best_idx = torch.argmax(masked_q).item()
    return self.idx_to_node.get(best_idx, None)
```
- **`torch.no_grad()`**: Disables gradient tracking (faster inference)
- **Masking**: Sets non-neighbor Q-values to `-inf` so `argmax` ignores them
- **`argmax`**: Finds index with highest Q-value
- **`.item()`**: Converts tensor to Python int

---

### Lines 102-104: Soft update helper
```python
def _soft_update(self, tau=0.01):
    for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
```
- **Polyak averaging**: Slowly moves target network toward main network
- **Formula**: `target = 0.01 * main + 0.99 * target`
- Stabilizes training (prevents oscillations)

---

### Line 106: `def learn(..., done=False):`
- **Experience replay training**

### Lines 110-111: Skip if PyTorch unavailable

### Lines 113-115: Store experience
```python
# Store transition in memory
self.memory.append((current_node, next_node, reward, next_state, next_neighbors, target_node, done))
```
- `done`: Boolean indicating if episode ended

### Lines 117-118: Wait for sufficient data
```python
if len(self.memory) < self.batch_size:
    return
```
- Need at least `batch_size` experiences before training

### Line 121: `batch = random.sample(self.memory, self.batch_size)`
- **Random sampling**: Breaks temporal correlations

### Lines 124-141: Prepare batch tensors
```python
states = []
next_states = []
actions = []
rewards = []
dones = []
valid_next_neighbor_masks = []

for (curr, act, rew, nxt, nxt_neigh, tgt, d) in batch:
    states.append(self._get_state_vector(curr, tgt))
    next_states.append(self._get_state_vector(nxt, tgt))
    actions.append(self.node_to_idx[act])
    rewards.append(rew)
    dones.append(1.0 if d else 0.0)
    neigh_indices = [self.node_to_idx[n] for n in nxt_neigh if n in self.node_to_idx]
    valid_next_neighbor_masks.append(neigh_indices)
```
- Converts batch to tensors for parallel processing

### Lines 143-147: Stack into tensors
```python
states_tensor = torch.stack(states)
next_states_tensor = torch.stack(next_states)
actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
```

### Lines 149-151: Compute predictions
```python
# 1. Prediction (Current State)
q_values = self.model(states_tensor)
q_pred = q_values.gather(1, actions_tensor)
```
- **`gather`**: Selects Q-values of actions that were taken

### Lines 153-169: Compute targets with target network
```python
# 2. Target (Next State) using Target Network
with torch.no_grad():
    next_q_values = self.target_model(next_states_tensor)
    
    # Compute Max Q for next state (constrained to neighbors)
    max_next_qs = []
    for i, neighbors_idx in enumerate(valid_next_neighbor_masks):
        if not neighbors_idx:
            max_next_qs.append(0.0)
        else:
            neighbor_qs = next_q_values[i, neighbors_idx]
            max_q = torch.max(neighbor_qs).item()
            max_next_qs.append(max_q)
    
    max_next_qs_tensor = torch.tensor(max_next_qs, dtype=torch.float32).unsqueeze(1)
    q_target = rewards_tensor + (self.gamma * max_next_qs_tensor * (1 - dones_tensor))
```
- **Target**: `r + γ * max_a' Q_target(s', a')`
- **`(1 - dones)`**: Zeros out future reward if episode ended

### Lines 171-178: Optimization step
```python
# 3. Optimization
loss = self.criterion(q_pred, q_target)
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()

# 4. Soft Update
self._soft_update()
```
- Standard PyTorch training loop
- Soft update after each batch

---

## GNN-RL Agent (Graph Neural Network)

### Line 181: `class GNNRLAgent:`
- Uses Graph Convolutional Network to embed nodes
- Captures topology structure

### Line 185: `def __init__(self, graph, nodes, ...embedding_dim=16...):`
- **New parameter**:
  - `embedding_dim=16`: Size of node embeddings after GCN

### Lines 198-200: Build adjacency matrix
```python
# Adjacency Matrix (static for now, can be updated)
self.adj_matrix = None 
self._build_adj_matrix()
```

### Lines 202-216: GCN architecture
```python
if TORCH_AVAILABLE:
    # GCN Layer: H_new = ReLU(A * H * W)
    # Input Features per node: [is_target, trust_score, degree_centrality]
    self.input_dim = 3 
    self.embedding_dim = embedding_dim
    
    # GCN weights
    self.gcn_weight = nn.Parameter(torch.randn(self.input_dim, self.embedding_dim))
    
    # Q-Net Head
    self.q_head = nn.Sequential(
        nn.Linear(2 * self.embedding_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
```
- **Input features**: 3D vector per node
- **GCN weight**: Learnable matrix for aggregation
- **Q-head**: MLP that takes edge embedding → Q-value

### Lines 218-225: Target network setup

### Lines 229-243: `_build_adj_matrix`
```python
def _build_adj_matrix(self):
    adj = torch.eye(self.num_nodes) # Self loops
    for u, v in self.graph.edges():
        if u in self.node_to_idx and v in self.node_to_idx:
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            adj[i, j] = 1.0
    
    # Normalize: D^-0.5 * A * D^-0.5
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    diag = torch.diag(deg_inv_sqrt)
    
    self.adj_matrix = torch.mm(torch.mm(diag, adj), diag)
```
- **Symmetric normalization**: Standard for GCNs
- Prevents vanishing/exploding gradients

### Lines 245-262: `_get_node_features`
```python
def _get_node_features(self, target_node):
    features = torch.zeros((self.num_nodes, self.input_dim))
    
    degrees = dict(self.graph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    
    for i, node in enumerate(self.nodes):
        # 1. Is Target
        if node == target_node:
            features[i, 0] = 1.0
        # 2. Trust 
        features[i, 1] = 1.0 
        # 3. Degree
        features[i, 2] = degrees.get(node, 0) / max_deg
        
    return features
```
- **Feature vector**: `[is_target, trust, normalized_degree]`
- Trust currently hardcoded to 1.0 (could be integrated)

### Lines 264-273: `_forward` - GCN propagation
```python
def _forward(self, target_node, use_target=False):
    """Returns Embeddings for all nodes"""
    X = self._get_node_features(target_node)
    A = self.adj_matrix
    W = self.target_gcn_weight if use_target else self.gcn_weight
    
    support = torch.mm(X, W)
    output = torch.mm(A, support)
    embeddings = F.relu(output)
    return embeddings
```
- **GCN formula**: `H = ReLU(A * X * W)`
  - `X`: Feature matrix `[N × 3]`
  - `W`: Weight matrix `[3 × 16]`
  - `A`: Normalized adjacency `[N × N]`
  - Output: `[N × 16]` embeddings

### Lines 275-311: `choose_action` with GCN
```python
with torch.no_grad():
    embeddings = self._forward(target_node, use_target=False)
    curr_emb = embeddings[self.node_to_idx[current_node]]
    
    best_n = None
    max_q = -float('inf')
    
    for n in neighbors:
        if avoid_nodes and n in avoid_nodes: continue
        if n not in self.node_to_idx: continue
        neigh_emb = embeddings[self.node_to_idx[n]]
        
        cat_emb = torch.cat([curr_emb, neigh_emb])
        q_val = self.q_head(cat_emb).item()
        
        if q_val > max_q:
            max_q = q_val
            best_n = n
    
    return best_n
```
- **Edge Q-value**: Function of `[current_emb || neighbor_emb]`
- Picks neighbor with highest Q

### Lines 320-372: `learn` method
- Similar to DQN but uses GCN for Q-value computation
- Batch training with target network and soft updates
