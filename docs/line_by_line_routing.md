# Routing Algorithms - Line-by-Line Explanation

## File: `routing.py`

### Line 1: `import networkx as nx`
- Imports NetworkX library for graph algorithms (Dijkstra, BFS, etc.)

### Line 3: `class RoutingAlgorithm:`
- Abstract base class for all routing algorithms
- Provides common interface for different routing strategies

### Line 4: `def __init__(self, graph):`
- Constructor accepting network graph
- **Parameter `graph`**: NetworkX DiGraph object

### Line 5: `self.graph = graph`
- Stores reference to network topology

### Line 7: `def find_path(self, source, target):`
- Abstract method to be implemented by subclasses
- **Parameter `source`**: Starting node ID
- **Parameter `target`**: Destination node ID
- **Returns**: List of nodes representing path, or None

### Line 8: `raise NotImplementedError("Subclasses must implement find_path")`
- Forces subclasses to override this method
- Raises error if called directly on base class

---

## Algorithm 1: Standard OSPF (Shortest Path)

### Line 10: `class ShortestPathRouting(RoutingAlgorithm):`
- Implements traditional Dijkstra's algorithm
- Finds path with minimum total weight (latency)
- **Trust-Unaware**: Ignores node reliability

### Line 11: `def find_path(self, source, target):`
- Overrides abstract method

### Line 12: `try:`
- Exception handling for cases where no path exists

### Line 13: `return nx.shortest_path(self.graph, source=source, target=target, weight='weight')`
- **`nx.shortest_path`**: NetworkX's Dijkstra implementation
- **`weight='weight'`**: Uses 'weight' attribute (latency) as cost metric
- **How it works**: Finds path minimizing sum of edge weights
- **Returns**: List of nodes like `[0, 2, 5, 9]`

### Line 14: `except nx.NetworkXNoPath:`
- Catches exception when source and target are disconnected

### Line 15: `return None`
- Returns None to indicate routing failure

---

## Algorithm 2: Intelligent Trust-Based Routing

### Line 17: `class IntelligentRouting(RoutingAlgorithm):`
- Advanced routing incorporating node trust scores
- Avoids untrusted nodes even if they offer shorter paths

### Line 18: `def __init__(self, graph, trust_model):`
- Constructor
- **Parameter `trust_model`**: TrustModel object with node reputation scores

### Line 19: `super().__init__(graph)`
- Calls parent class constructor to store graph

### Line 20: `self.trust_model = trust_model`
- Stores reference to trust scoring system

### Line 22: `def calculate_cost(self, u, v, data):`
- **Custom cost function** for pathfinding
- Called by NetworkX for each edge during Dijkstra
- **Parameter `u`**: Source node of edge
- **Parameter `v`**: Destination node of edge
- **Parameter `data`**: Edge attributes dictionary

### Line 23-27: `""" Custom cost function that considers: - Link latency (weight) - Destination Node Trust (or next hop trust) """`
- Docstring explaining dual-metric approach

### Line 28: `base_cost = data.get('weight', 1)`
- Extracts physical latency from edge
- Defaults to 1 if missing

### Line 29: `trust_score = self.trust_model.get_trust(v)`
- Gets trust score of **destination node** `v`
- Range: 0.0 (untrusted) to 1.0 (fully trusted)

### Line 31-35: Comments explaining cost formula
- `# Invert trust so lower trust = higher cost`
- `# cost = base_weight * (1 + (1 - trust))`
- `# If trust is 1.0, cost = base_weight`
- `# If trust is 0.0, cost = base_weight * 2 (or more extreme penalty)`

### Line 35: `trust_penalty = (1 - trust_score) * 10 # Multiplier for impact`
- Calculates penalty based on lack of trust
- **Formula breakdown**:
  - If trust = 1.0 → penalty = 0
  - If trust = 0.5 → penalty = 5
  - If trust = 0.0 → penalty = 10

### Line 37: `return base_cost * (1 + trust_penalty)`
- **Final cost formula**: `Latency × (1 + (1 - Trust) × 10)`
- **Examples**:
  - Trusted node (trust=1.0, latency=10ms): Cost = 10 × 1 = 10
  - Neutral node (trust=0.5, latency=10ms): Cost = 10 × 6 = 60
  - Untrusted node (trust=0.0, latency=10ms): Cost = 10 × 11 = 110
- **Effect**: Makes bad nodes "look" 11× farther away

### Line 39: `def find_path(self, source, target):`
- Path finding method

### Line 40: `try:`
- Exception handling

### Line 41: `return nx.shortest_path(self.graph, source=source, target=target, weight=self.calculate_cost)`
- Uses Dijkstra but with **custom weight function**
- **`weight=self.calculate_cost`**: Passes function instead of attribute name
- NetworkX calls `calculate_cost` for each edge during search

### Line 42: `except nx.NetworkXNoPath:`
- Handles disconnected case

### Line 43: `return None`
- Returns None if no path exists

---

## Algorithm 3: Reinforcement Learning Routing

### Line 45: `class RLRouting(RoutingAlgorithm):`
- AI-based routing using RL agents
- Makes hop-by-hop decisions without global topology knowledge

### Line 46: `def __init__(self, graph, agent):`
- Constructor
- **Parameter `agent`**: Could be `QLearningAgent`, `DQNAgent`, or `GNNRLAgent`

### Line 47: `super().__init__(graph)`
- Stores graph reference

### Line 48: `self.agent = agent`
- Stores reference to AI agent

### Line 50: `def find_path(self, source, target):`
- Path finding method

### Line 51-55: `""" Route packet hop-by-hop using Q-Learning Agent. Note: This is different from Dijkstra. We don't return a full path upfront typically in RL routing, but for this simulation compatibility, we will simulate the hop-by-hop decisions to generate a 'path'. """`
- Docstring explaining RL approach difference

### Line 56: `path = [source]`
- Initializes path with starting node

### Line 57: `current = source`
- Tracks current position

### Line 58: `visited = set([source])`
- Tracks visited nodes to prevent loops

### Line 59: `max_hops = 20`
- Safety limit to prevent infinite loops

### Line 61: `while current != target and len(path) < max_hops:`
- Continues until destination reached or hop limit exceeded

### Line 62: `neighbors = list(self.graph.neighbors(current))`
- Gets valid next hops (outgoing edges from current node)

### Line 63-64: `if not neighbors:` `break`
- Dead end: No outgoing edges, routing fails

### Line 66-72: Comments about `avoid_nodes` parameter
- `# Pass visited nodes to exclude loops`
- Explains that some agents support loop prevention

### Line 73: `if "avoid_nodes" in self.agent.choose_action.__code__.co_varnames:`
- **Python introspection**: Checks if agent's `choose_action` method accepts `avoid_nodes` parameter
- Uses `__code__.co_varnames` to inspect function signature

### Line 74: `next_hop = self.agent.choose_action(current, neighbors, target, avoid_nodes=visited)`
- Calls agent with loop prevention
- Agent uses AI policy to select best neighbor

### Line 75: `else:`
- Agent doesn't support `avoid_nodes`

### Line 76: `next_hop = self.agent.choose_action(current, neighbors, target)`
- Calls without loop prevention parameter

### Line 78-79: `if next_hop is None:` `break`
- Agent couldn't decide (e.g., all neighbors visited)

### Line 81-87: Comments about loop handling
- `# Prevent simple loops`
- Discusses trade-offs of allowing revisits

### Line 89: `path.append(next_hop)`
- Adds chosen hop to path

### Line 90: `visited.add(next_hop)`
- Marks node as visited

### Line 91: `current = next_hop`
- Moves to next node

### Line 93: `if current == target:`
- Checks if destination reached

### Line 94: `return path`
- Returns successful path

### Line 95: `return None`
- Failed to reach destination

---

## Algorithm 4: RIP (Hop Count)

### Line 97: `class RIPRouting(RoutingAlgorithm):`
- Simulates RIP (Routing Information Protocol)

### Line 98-101: `""" Simulates RIP (Routing Information Protocol). - Metric: Hop Count (Distance Vector). - Ignores Link Latency/Congestion. """`
- Docstring explaining RIP characteristics

### Line 103: `def __init__(self, graph):`
- Simple constructor

### Line 104: `self.graph = graph`
- Stores graph

### Line 106: `def find_path(self, source, target):`
- Path finding method

### Line 107: `try:`
- Exception handling

### Line 108: `# Shortest path with weight=None implies BFS (Hop Count)`
- Comment explaining how RIP works

### Line 109: `return nx.shortest_path(self.graph, source=source, target=target, weight=None)`
- **`weight=None`**: Treats all edges equally
- **Effect**: Finds path with minimum number of hops
- **Ignores**: Latency, trust, congestion
- **Uses**: Breadth-First Search (BFS) internally

### Line 110: `except nx.NetworkXNoPath:`
- Handles disconnected case

### Line 111: `return None`
- Returns None if no path

---

## Summary of Routing Algorithms

| Algorithm | Metric | Trust-Aware | Adaptive |
|-----------|--------|-------------|----------|
| **Standard OSPF** | Latency only | ❌ No | ❌ No |
| **Intelligent** | Latency + Trust | ✅ Yes | ✅ Yes (learns from drops) |
| **RL (Q/DQN/GNN)** | Learned reward | ✅ Implicit | ✅ Yes (self-learning) |
| **RIP** | Hop count | ❌ No | ❌ No |
