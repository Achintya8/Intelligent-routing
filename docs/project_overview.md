# Intelligent Routing Project Overview

## 1. Project Purpose
The **Intelligent Routing Project** is a sophisticated network simulation designed to demonstrate and compare different routing protocols in a dynamic, potentially hostile network environment. Its primary goal is to showcase how **AI-driven** and **Trust-based** routing algorithms can outperform traditional static protocols (like Dijkstra/OSPF) when faced with:
*   **Packet Loss & Congestion**: Dynamic latency spikes and link failures.
*   **Adversarial Attacks**: Malicious nodes dropping packets (Blackhole, Grayhole, On-Off attacks).
*   **QoS Requirements**: Prioritizing Voice traffic over Data traffic.

## 2. High-Level Architecture
The project is built around a discrete-event simulation using `SimPy` and graph theory using `NetworkX`.

### Core Components
1.  **Network Environment (`network_sim.py`)**:
    *   Manages the simulation time and events.
    *   Defines the network topology (nodes and edges) using `NetworkX`.
    *   Simulates physical link properties like **latency** and **bandwidth**.
    *   Handles **stochastic packet drops** based on congestion and node reliability.

2.  **Routing Logic (`routing.py`)**:
    *   **Shortest Path (Standard)**: Uses Dijkstra's algorithm to find the path with minimum static weight (latency). It assumes all nodes are trustworthy.
    *   **Intelligent Routing**: Incorporates a **Trust Model**. It calculates a "cost" for each link that combines latency with the destination's trust score. If a node has low trust, the cost is artificially inflated, causing the algorithm to route around it.
    *   **RL Routing**: Uses Reinforcement Learning agents to make hop-by-hop routing decisions without knowing the full global topology.

3.  **Agents & AI (`rl_agent.py`, `advanced_agents.py`)**:
    *   **Q-Learning**: A tabular RL approach where nodes learn `Q(state, action)` values. `State` is the current node, `Action` is the next hop.
    *   **DQN (Deep Q-Network)**: Uses a Neural Network to approximate Q-values, allowing it to handle larger state spaces (input: One-hot vectors of Source + Target).
    *   **GNN-RL (Graph Neural Network)**: Uses a Graph Neural Network to generate node embeddings that capture the topology structure. This allows the agent to "see" the graph structure and make smarter decisions based on neighbors' features (Trust, Degree).

4.  **Trust & Security (`trust_model.py`, `security.py`)**:
    *   **Trust Model**: Maintains a trust score (0.0 to 1.0) for each node.
        *   **Reward**: Successful packet delivery increases trust (+0.05).
        *   **Penalty**: Packet loss decreases trust (x0.95 or similar).
    *   **Adversaries**: Simulates specific attack patterns:
        *   **Blackhole**: Drops 100% of packets.
        *   **Grayhole**: Drops specific packet types (e.g., Data) but forwards others (Voice) to evade detection.
        *   **On-Off**: Alternates between good and bad behavior to trick the trust model.

5.  **Visualization & UI (`dashboard.py`, `visualization.py`)**:
    *   **Streamlit Dashboard**: A web-based UI to run simulations interactively.
    *   **Real-time Metrics**: Displays Packet Delivery Ratio (PDR) and Latency.
    *   **Network Graph**: Visualizes the topology, analyzing trust scores (Green = Trusted, Red = Untrusted).

## 3. Key Workflows

### A. The Simulation Loop (`main.py`)
1.  **Setup**: Initialize `SimPy` environment and create `NetworkX` graph.
2.  **Traffic Generation**: Randomly select Source-Destination pairs.
3.  **Routing**: The selected Algorithm (`ShortestPath`, `Intelligent`, `RL`) determines the path.
4.  **Execution**: The packet traverses the path hop-by-hop.
    *   At each hop, the `NetworkSimulation` checks for drops (Congestion or Malicious reliability).
    *   If dropped, the `TrustModel` penalizes the node.
    *   If delivered, the `TrustModel` rewards the nodes.
5.  **Feedback (RL)**: If using RL, the agent receives a Reward (Positive for success, Negative for failure/latency) and updates its policy.

### B. Comparative Analysis (`compare_algos.py`)
Runs head-to-head battles between algorithms on the *exact same* network conditions to generate performance charts. It proves that Intelligent/RL routing maintains >90% PDR while Standard routing fails (~60%) in hostile networks.

### C. QoS Simulation (`qos_sim.py`)
Demonstrates traffic differentiation. "Voice" packets (High Priority) are allowed through congested links where "Data" packets (Low Priority) would be dropped.

## 4. Technology Stack
*   **Language**: Python 3.9+
*   **Simulation**: `SimPy` (Discrete Event Simulation)
*   **Graph Theory**: `NetworkX` (Pathfinding, Topology)
*   **AI/ML**: `PyTorch` (DQN, GNN), `NumPy`
*   **Visualization**: `Streamlit` (Dashboard), `Matplotlib` (Charts)
