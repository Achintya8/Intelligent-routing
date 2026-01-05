# Core Simulation Files - Line-by-Line Explanation

## File 1: `utils.py`

### Line 1: `import logging`
- Imports Python's built-in logging module for structured log output

### Line 3: `def setup_logger(name='NetworkSim'):`
- Defines a function to create and configure a logger
- **Parameter `name`**: Logger identifier (default: 'NetworkSim')

### Line 4: `logger = logging.getLogger(name)`
- Creates or retrieves a logger instance with the specified name

### Line 5: `logger.setLevel(logging.DEBUG)`
- Sets minimum logging level to DEBUG (most verbose)
- Allows all message types: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Line 6: `if not logger.handlers:`
- Checks if the logger already has handlers attached
- Prevents duplicate handlers if function is called multiple times

### Line 7: `ch = logging.StreamHandler()`
- Creates a console/stream handler (outputs to terminal/console)

### Line 8: `ch.setLevel(logging.DEBUG)`
- Sets handler's minimum level to DEBUG

### Line 9: `formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')`
- Defines log message format:
  - `%(asctime)s`: Timestamp
  - `%(name)s`: Logger name
  - `%(levelname)s`: Severity (DEBUG/INFO/etc.)
  - `%(message)s`: Actual log message

### Line 10: `ch.setFormatter(formatter)`
- Applies the formatter to the handler

### Line 11: `logger.addHandler(ch)`
- Attaches the configured handler to the logger

### Line 12: `return logger`
- Returns the configured logger object for use

---

## File 2: `network_sim.py`

### Line 1: `import simpy`
- Imports SimPy: Discrete-event simulation framework

### Line 2: `import networkx as nx`
- Imports NetworkX: Graph theory and network analysis library

### Line 3: `import random`
- Imports random module for stochastic behavior

### Line 4: `from utils import setup_logger`
- Imports our custom logger setup function

### Line 6: `logger = setup_logger()`
- Initializes logger for this module

### Line 8: `class NetworkSimulation:`
- Defines the main network simulation class

### Line 9: `def __init__(self, env):`
- Constructor method
- **Parameter `env`**: SimPy Environment object (manages simulation time)

### Line 10: `self.env = env`
- Stores reference to simulation environment

### Line 11: `self.graph = nx.DiGraph()`
- Creates empty directed graph (DiGraph allows one-way links)

### Line 12: `self.nodes = []`
- Initializes empty list to store node IDs

### Line 14: `def create_topology(self, num_nodes=10, connectivity=0.3):`
- Method to generate random network topology
- **Parameter `num_nodes`**: Number of network nodes (default: 10)
- **Parameter `connectivity`**: Probability of edge between any two nodes (0.0-1.0)

### Line 15: `"""Randomly generates a network topology"""`
- Docstring describing the method

### Line 16: `self.graph = nx.gnp_random_graph(num_nodes, connectivity, directed=True)`
- Uses Erdős–Rényi G(n,p) model to create random graph
- Each possible edge exists with probability `connectivity`
- `directed=True`: Creates directed graph (A→B ≠ B→A)

### Line 17: `# Ensure weights (latency) exist`
- Comment explaining next loop purpose

### Line 18: `for (u, v) in self.graph.edges():`
- Iterates over all edges in the graph
- `u`: Source node, `v`: Destination node

### Line 19: `self.graph.edges[u, v]['weight'] = random.randint(1, 10) # ms latency`
- Assigns random edge weight (1-10 milliseconds)
- This represents link latency/delay

### Line 20: `self.graph.edges[u, v]['capacity'] = random.randint(10, 100) # Mbps`
- Assigns random capacity (10-100 Mbps)
- Represents link bandwidth

### Line 22: `self.nodes = list(self.graph.nodes())`
- Converts graph nodes to list and stores

### Line 23: `logger.info(f"Topology created with {num_nodes} nodes and {len(self.graph.edges())} edges")`
- Logs topology creation with node and edge counts

### Line 25: `def degrade_node(self, node_id, duration=50):`
- Method to simulate node performance issues
- **Parameter `node_id`**: ID of node to degrade
- **Parameter `duration`**: How long degradation lasts

### Line 26: `"""Simulates a node having performance issues or being compromised"""`
- Docstring

### Line 27: `logger.warning(f"Node {node_id} is degrading...")`
- Logs warning about node degradation

### Line 29: `def update_congestion(self):`
- Method to dynamically update network congestion

### Line 30-32: `""" Periodically updates edge weights to simulate congestion. Run this as a SimPy process. """`
- Docstring explaining this is a SimPy generator process

### Line 34: `while True:`
- Infinite loop (runs throughout simulation)

### Line 35: `yield self.env.timeout(5) # Every 5 ticks`
- Pauses execution for 5 simulation time units
- `yield` makes this a generator (SimPy process requirement)

### Line 37-38: `# Select random edges to congest`
`if self.graph.edges:`
- Checks if graph has any edges

### Line 39: `u, v = random.choice(list(self.graph.edges()))`
- Randomly selects one edge from the graph

### Line 40: `current_weight = self.graph[u][v]['weight']`
- Retrieves current latency of selected edge

### Line 42-43: `# Randomly spike latency or recover`
`if random.random() < 0.3:`
- 30% chance of congestion spike

### Line 44-45: `# Congestion Spike (Cap at 200ms)`
`new_weight = min(200, current_weight + random.randint(10, 50))`
- Increases latency by 10-50ms
- Caps maximum at 200ms (prevents infinite growth)

### Line 46: `logger.debug(f"Congestion on link {u}->{v}: Weight {current_weight} -> {new_weight}")`
- Logs the congestion event

### Line 47: `else:`
- 70% chance of recovery

### Line 48-49: `# Recovery (decay back to baseline 1-10)`
`new_weight = max(random.randint(1, 10), current_weight - 10)`
- Decreases latency by 10ms or resets to baseline (1-10ms)
- Uses `max` to prevent going below baseline

### Line 51: `self.graph[u][v]['weight'] = new_weight`
- Updates the edge weight in the graph

### Line 53: `def simulate_packet(self, path, trust_model, priority=0):`
- Simulates packet traversal along a path
- **Parameter `path`**: List of nodes representing the route
- **Parameter `trust_model`**: TrustModel object (can be None)
- **Parameter `priority`**: 0 (Data/Low) or 1 (Voice/High)

### Line 54-56: `""" Simulates a packet traversing a path. priority: 0 (Normal/Data), 1 (High/Voice) """`
- Docstring explaining priority levels

### Line 58-59: `if not path:` `return False`
- If no path exists, packet fails immediately

### Line 61: `success = True`
- Assumes success initially

### Line 62: `for i in range(len(path) - 1):`
- Iterates through each hop in the path
- `len(path) - 1` because each iteration checks edge i→i+1

### Line 63: `u, v = path[i], path[i+1]`
- `u`: Current node, `v`: Next hop node

### Line 65-66: `# 1. Congestion Check`
`# If link is highly congested (weight > 50), Low Priority packets might get dropped`
- Comment explaining QoS logic

### Line 67: `weight = self.graph[u][v].get('weight', 1)`
- Gets edge latency (default 1 if missing)

### Line 68: `if weight > 50 and priority == 0:`
- Checks if link is congested AND packet is low priority

### Line 69-70: `# 30% drop chance for Data during congestion`
`if random.random() < 0.3:`
- 30% probability of dropping low-priority packets

### Line 71: `logger.debug(f"Packet (Low Prio) dropped due to congestion on {u}->{v}")`
- Logs the congestion drop

### Line 72: `return False`
- Packet fails, returns immediately

### Line 74-75: `# ... Rest of logic usually handled by "realistic_simulate_packet" patch`
`# But the base method also needs to be robust if called directly`
- Comment noting monkey-patching in main.py

### Line 77: `u, v = path[i], path[i+1]`
- Redundant line (already done on line 63)

### Line 79-80: `# Simple stochastic model for packet loss`
`# Lower trust = higher chance of drop`
- Comment explaining trust-based dropping

### Line 81: `trust = trust_model.get_trust(v)`
- Retrieves trust score of destination node `v`

### Line 82: `drop_prob = (1 - trust) * 0.5 # Max 50% drop rate for 0 trust`
- Calculates drop probability based on trust
- Formula: If trust=0 → 50% drop, if trust=1 → 0% drop

### Line 84: `if random.random() < drop_prob:`
- Randomly determines if packet is dropped

### Line 85: `logger.info(f"Packet dropped at node {v} (Trust: {trust:.2f})")`
- Logs the trust-based drop

### Line 86: `success = False`
- Marks packet as failed

### Line 87: `trust_model.update_trust(v, False) # Penalize`
- Decreases trust of node that dropped packet

### Line 88: `break`
- Exits loop (packet failed)

### Line 89: `else:`
- Packet successfully passed this hop

### Line 90: `trust_model.update_trust(v, True) # Reward`
- Increases trust of reliable node

### Line 92: `return success`
- Returns final success/failure status

---

## File 3: `trust_model.py`

### Line 1: `class TrustModel:`
- Defines the trust management class

### Line 2: `def __init__(self, initial_trust=1.0, decay_factor=0.95, bonus_factor=0.05):`
- Constructor for trust model
- **Parameter `initial_trust`**: Starting trust for all nodes (default: 1.0 = fully trusted)
- **Parameter `decay_factor`**: Multiplier for failures (default: 0.95)
- **Parameter `bonus_factor`**: Addition for successes (default: 0.05)

### Line 3: `self.node_trust = {}`
- Dictionary to store trust scores: {node_id: trust_score}

### Line 4: `self.initial_trust = initial_trust`
- Stores initial trust value

### Line 5: `self.decay_factor = decay_factor # Penalty for bad behavior`
- Stores decay factor (multiplicative decrease)

### Line 6: `self.bonus_factor = bonus_factor # Bonus for good behavior`
- Stores bonus factor (additive increase)

### Line 8: `def initialize_node(self, node_id):`
- Method to initialize a node's trust score

### Line 9: `if node_id not in self.node_trust:`
- Checks if node is already initialized

### Line 10: `self.node_trust[node_id] = self.initial_trust`
- Sets node trust to initial value (typically 1.0)

### Line 12: `def update_trust(self, node_id, success):`
- Updates trust based on transaction outcome
- **Parameter `node_id`**: Node to update
- **Parameter `success`**: Boolean (True if packet delivered, False if dropped)

### Line 13-15: `""" Updates trust score based on transaction success/failure. success: bool (True if packet delivered successfully, False otherwise) """`
- Docstring

### Line 17-18: `if node_id not in self.node_trust:` `self.initialize_node(node_id)`
- Lazy initialization: Creates entry if node is new

### Line 20: `current = self.node_trust[node_id]`
- Retrieves current trust score

### Line 21: `if success:`
- If packet was successfully delivered

### Line 22-23: `# Increase trust, capped at 1.0`
`self.node_trust[node_id] = min(1.0, current + self.bonus_factor)`
- **Additive Increase**: Adds 0.05 to trust
- **Capped**: Maximum trust is 1.0
- **Effect**: Slow recovery (20 successes to go from 0 to 1)

### Line 24: `else:`
- If packet was dropped

### Line 25-26: `# Decrease trust`
`self.node_trust[node_id] = max(0.0, current * self.decay_factor)`
- **Multiplicative Decrease**: Multiplies by 0.95
- **Effect**: Fast penalty (14 failures to drop from 1.0 to 0.5)
- **Capped**: Minimum trust is 0.0

### Line 28: `def get_trust(self, node_id):`
- Retrieves trust score for a node

### Line 29: `return self.node_trust.get(node_id, self.initial_trust)`
- Returns stored trust or initial value (1.0) if node is unknown
- Uses dictionary `.get()` method for safe access
