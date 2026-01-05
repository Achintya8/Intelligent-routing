# Codebase Explanation

This document provides a detailed, line-by-line explanation of the key files in the Intelligent Routing project.

---

## 1. Network Simulation (`network_sim.py`)
This file defines the physical environment of the network.

*   **Imports**: `simpy` for event scheduling, `networkx` for graph structure, `random` for stochastic behavior.
*   **`NetworkSimulation` Class**:
    *   `__init__(env)`: Initializes the SimPy environment and an empty NetworkX DiGraph (Directed Graph).
    *   `create_topology(num_nodes, connectivity)`:
        *   Uses `nx.gnp_random_graph` to create a random graph.
        *   Iterates over edges to assign `weight` (latency in ms) and `capacity` (bandwidth).
    *   `update_congestion()`:
        *   A **SimPy Process** (generator) that runs indefinitely (`while True`).
        *   Every 5 time units (`yield env.timeout(5)`), it picks a random edge.
        *   It updates the edge `weight` (latency).
        *   **Simulates Congestion**: 30% chance to spike latency (up to 200ms).
        *   **Simulates Recovery**: 70% chance to reduce latency back to normal levels (1-10ms).
    *   `simulate_packet(path, trust_model, priority)`:
        *   Takes a calculated `path` and attempts to traverse it.
        *   **Congestion Drop**: If an edge weight is > 50 (congested) AND priority is 0 (Low/Data), there is a 30% chance the packet is dropped immediately.
        *   **Reliability Drop**: (Usually handled by patches in `main.py` or `security.py`, but base logic exists here). Checks `trust_model.get_trust(v)`. If trust is low, drop probability increases.
        *   **Trust Update**:
            *   If dropped: Call `trust_model.update_trust(v, False)`.
            *   If passed: Call `trust_model.update_trust(v, True)`.

---

## 2. Routing Logic (`routing.py`)
Defines the algorithms used to find paths.

*   **`RoutingAlgorithm` (Base Class)**: Abstract base class enforcing `find_path` interface.
*   **`ShortestPathRouting`**:
    *   `find_path`: Calls `nx.shortest_path` using `weight='weight'`. This is standard Dijkstra, optimizing for lowest latency but ignoring security/trust.
*   **`IntelligentRouting`**:
    *   `__init__`: Accepts a `trust_model`.
    *   `calculate_cost`: A custom function passed to NetworkX.
        *   `base_cost`: The physical link latency.
        *   `trust_score`: Get trust of destination node.
        *   **Formula**: `Cost = Latency * (1 + (1 - Trust) * 10)`.
        *   **Effect**: If a node is untrusted (0.5), cost increases by 5x (or more), making Dijkstra avoid it unless it's the *only* path.
    *   `find_path`: Calls `nx.shortest_path` using the custom `calculate_cost` function.
*   **`RLRouting`**:
    *   Wrapper for reinforcement learning agents.
    *   `find_path`: DOes not find a full path at once. Instead, it simulates hop-by-hop traversal.
    *   Loop: `current` starts at `source`.
    *   Calls `agent.choose_action` to pick next neighbor.
    *   Updates `current` node until `target` is reached or `max_hops` exceeded.
*   **`RIPRouting`**:
    *   Simulates RIP (Routing Information Protocol).
    *   `find_path`: Calls `nx.shortest_path` with `weight=None`. This effectively performs BFS (Breadth-First Search), minimizing **Hop Count** regardless of latency or trust.

---

## 3. Trust Model (`trust_model.py`)
Manages the reputation of nodes.

*   **`TrustModel` Class**:
    *   `__init__`: Sets defaults (`initial_trust=1.0`, `decay=0.95`, `bonus=0.05`).
    *   `update_trust(node_id, success)`:
        *   **Success**: `trust = min(1.0, current + 0.05)`. Slowly builds trust.
        *   **Failure**: `trust = max(0.0, current * 0.95)`. Exponentially decreases trust (Multiplicative Decrease).
    *   `get_trust`: Returns current score or 1.0 if unknown.

---

## 4. Reinforcement Learning Agents

### Basic Agent (`rl_agent.py`)
*   **`QLearningAgent`**:
    *   `__init__`: Initializes a Q-Table (Dictionary of Dictionaries: `Q[state][action]`).
    *   `choose_action(current, neighbors)`:
        *   **Epsilon-Greedy**: 
            *   `random() < epsilon`: Explore (Pick random neighbor).
            *   Else: Exploit (Pick neighbor with highest Q-value).
    *   `learn(state, action, reward, next_state, ...)`:
        *   **Bellman Update**: `Q(s,a) = Q(s,a) + alpha * (Reward + gamma * max(Q(s')) - Q(s,a))`.
        *   Updates the table based on the reward received.

### Advanced Agents (`advanced_agents.py`)
*   **`DQNAgent`**:
    *   Uses **PyTorch**.
    *   **Model**: A Feed-Forward Neural Network (`nn.Sequential`).
        *   **Input**: One-hot vector of (Current Node + Target Node). Size `2 * num_nodes`.
        *   **Output**: Q-values for *all* possible nodes.
    *   **Masking**: In `choose_action`, it masks outputs corresponding to non-neighbors (sets them to `-inf`) so the agent only picks valid physical links.
    *   **Replay Buffer (`deque`)**: Stores experiences `(state, action, reward, next_state)` to train in batches, stabilizing learning.
    *   **Target Network**: A copy of the main model, updated slowly (`soft_update`), to prevent training oscillations.
*   **`GNNRLAgent`**:
    *   Uses **Graph Neural Networks (GCN)**.
    *   **Features**: Inputs per node are `[Is_Target, Trust_Score, Centrality]`.
    *   **GCN Layer**: Aggregates information from neighbors using the Adjacency Matrix (`A`). `H_new = ReLU(A * H * W)`.
    *   **Q-Head**: A small NN that takes `[Current_Node_Embedding + Neighbor_Embedding]` and outputs a Q-value score for that edge.
    *   **Benefit**: This agent "sees" the topology. If a neighbor is good but connected to a bad node, the GNN embedding might reflect that "badness" from further hops.

---

## 5. Security & Attacks (`security.py`)
*   **`Adversary` Class**:
    *   `__init__`: Sets attack type (`blackhole`, `grayhole`, `on-off`).
    *   `process_packet(type)`: Decides whether to drop a packet.
        *   **Blackhole**: Returns `False` (Drop) always.
        *   **Grayhole**: Drops 'data', forwards 'voice'.
    *   `update_behavior`: (SimPy process)
        *   **On-Off Attack**: Cycles state between `good` (forwarding) and `bad` (dropping) every 20s/5s to confuse the Trust Model.

---

## 6. Simulation Runners

### Main Simulation (`main.py`)
*   Sets up the environment.
*   **Monkey Patching**: Defines `realistic_simulate_packet` to inject "Real" node reliability (ground truth) separate from the "Trust Model" (agent belief). This allows testing if the Trust Model *learns* the truth.
*   Runs comparison:
    1.  **Standard**: Runs Dijkstra. Logs results.
    2.  **Intelligent**: Runs Trust-based. Logs results.
    3.  **RL**: Trains Q-Learning agent (500 packets), then tests it.
*   Uses `visualization.py` to save `network_initial.png` and `network_final_trust.png`.

### Comparative Analysis (`compare_algos.py`)
*   Systematic benchmark.
*   `run_scenario`: Encapsulates setup, traffic generation, and metrics collection for one algo.
*   Generates `comparison_chart.png` using Maplotlib, plotting PDR and Latency side-by-side.

### Dashboard (`dashboard.py`)
*   **Streamlit Application**.
*   **Session State**: Persists `env`, `net_sim`, and `agents` across re-runs so the RL agents continue learning.
*   **Tabs**:
    *   **Live Simulation**: Allows stepping through packets one by one or in batches. Shows the Network Graph (updating in real-time).
    *   **Benchmark**: Runs the `compare_algos` logic interactively inside the browser.

---

## 7. Visualization (`visualization.py`)
*   `visualize_network`:
    *   Uses `nx.spring_layout` for node positioning (seeded for consistency).
    *   **Coloring**: Iterates nodes. If `trust < 0.5` (or `reliability < 0.8`), paints **RED**. Else **GREEN**.
    *   Draws edges with weights (latency) labeled.
