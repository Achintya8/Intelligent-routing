# Simulation Scripts - Line-by-Line Explanation

## File 1: `main.py` - Main Simulation Runner

This file runs comparative tests of the three main algorithms.

### Lines 1-10: Duplicate imports (lines 1-5 and 6-10 are identical)
```python
import simpy
import networkx as nx
import random
import logging
from utils import setup_logger
```
- Note: Lines duplicated (likely copy-paste error)

### Lines 11-14: Module imports
```python
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import ShortestPathRouting, IntelligentRouting, RLRouting
from rl_agent import QLearningAgent
```

### Line 16: `logger = setup_logger("Main")`

---

### Line 18: `def run_simulation(env, net_sim, trust_model, routing_algo, num_packets=50):`
- **Core simulation function**
- Generates traffic and tests routing algorithm
- **Returns**: Success count

### Lines 22-23: Docstring and initialization
```python
"""
Generates traffic and attempts to route it using the given algorithm.
"""
logger.info(f"Starting simulation with {routing_algo.__class__.__name__}")
```

### Lines 24-25: Statistics tracking
```python
success_count = 0
total_latency = 0
```

### Lines 27-33: Traffic generation
```python
# Simple traffic generator
# We will pick 10 random source-dest pairs/flows
flows = []
nodes = net_sim.nodes
for _ in range(10):
    src, dst = random.sample(nodes, 2)
    flows.append((src, dst))
```
- Creates 10 random (source, destination) pairs
- Realistic: Simulates common communication patterns

### Line 35: `for i in range(num_packets):`
- Main packet loop

### Line 36: `src, dst = random.choice(flows)`
- Randomly selects one flow per packet

### Lines 38-43: Routing decision
```python
# 1. Routing Decision
path = routing_algo.find_path(src, dst)

if not path:
    logger.debug(f"Packet {i}: No path found from {src} to {dst}")
    continue
```

### Lines 45-47: Calculate latency
```python
# 2. Simulate Transmission
# Calculate theoretical latency based on path weights
path_latency = sum(net_sim.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
```
- **`zip(path[:-1], path[1:])`**: Creates edge pairs
- Example: `[0,2,5,9]` → `[(0,2), (2,5), (5,9)]`

### Lines 49-54: Simulate and update stats
```python
# 3. Simulate success/failure (Trust/Reliability check)
success = net_sim.simulate_packet(path, trust_model)

if success:
    success_count += 1
    total_latency += path_latency
```

### Lines 56-73: RL feedback loop
```python
# 4. Feedback Loop for RL (if applicable)
if isinstance(routing_algo, RLRouting):
    # Reward formulation:
    # - High penalty for failure (-100)
    # - Negative latency for success (e.g., -Latency) to minimize it
    
    reward = -path_latency if success else -100
    
    # Update Q-Values along the path
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        next_neighbors = list(net_sim.graph.neighbors(v)) if v in net_sim.graph else []
        routing_algo.agent.learn(u, v, reward, v, next_neighbors)
```
- **Reward shaping**:
  - Success: `-latency` (encourages faster paths)
  - Failure: `-100` (heavy penalty)
- Updates Q-values for every hop in the path

### Lines 75-82: Logging and return
```python
yield env.timeout(random.uniform(0.1, 0.5))

logger.info(f"--- Results for {routing_algo.__class__.__name__} ---")
logger.info(f"Total Packets: {num_packets}")
logger.info(f"Success: {success_count}")
logger.info(f"Avg Latency (of successful): {total_latency/success_count if success_count else 0:.2f}")
return success_count
```
- `yield`: Makes this a SimPy process
- Logs final metrics

---

### Lines 84-217: `main()` function
Orchestrates three simulation runs.

### Lines 86-92: **Setup**
```python
env = simpy.Environment()
net_sim = NetworkSimulation(env)
random.seed(42)
net_sim.create_topology(num_nodes=15, connectivity=0.2)
```
- **Fixed seed (42)**: Ensures reproducibility

### Lines 94-114: **Ground truth setup (monkey patching)**
```python
# Introduce some "Bad" nodes
bad_nodes = [3, 7]
for n in net_sim.nodes:
    net_sim.graph.nodes[n]['reliability'] = 0.98 # Good nodes

for n in bad_nodes:
    if n in net_sim.graph.nodes:
        net_sim.graph.nodes[n]['reliability'] = 0.6 # Bad nodes
        logger.info(f"Setting Node {n} as Unreliable (60%)")
```
- **Separates**: "Real" reliability (ground truth) from Trust Model (belief)

### Lines 116-134: **Monkey patch `simulate_packet`**
```python
def realistic_simulate_packet(path, trust_model):
    if not path: return False
    success = True
    for i in range(len(path) - 1):
         u, v = path[i], path[i+1]
         reliability = net_sim.graph.nodes[v].get('reliability', 1.0)
         if random.random() > reliability:
             logger.debug(f"Packet dropped at REAL bad node {v}")
             success = False
             if trust_model:
                 trust_model.update_trust(v, False) # Algorithm learns
             break
         else:
             if trust_model:
                 trust_model.update_trust(v, True)
    return success

net_sim.simulate_packet = realistic_simulate_packet
```
- **Overrides**: Default packet simulation
- Uses `reliability` attribute instead of trust for drops
- **Allows**: Trust model to *learn* ground truth

### Lines 136-138: Visualization
```python
from visualization import visualize_network
visualize_network(net_sim.graph, filename="network_initial.png")
```

### Lines 140-150: **Run 1: Standard Dijkstra**
```python
print("\n--- Simulation 1: Standard Dijkstra ---")

env.process(net_sim.update_congestion())

trust_model_std = TrustModel() # Reset trust
routing_std = ShortestPathRouting(net_sim.graph)
proc1 = env.process(run_simulation(env, net_sim, trust_model_std, routing_std, num_packets=50))
env.run(until=proc1)
```
- Starts congestion process
- Runs 50 packets
- **Expected**: Poor performance (routes through bad nodes)

### Lines 152-180: **Run 2: Intelligent Routing**
```python
print("\n--- Simulation 2: Intelligent Routing ---")
env2 = simpy.Environment() # New time

# Re-create net_sim for fair comparison
net_sim2 = NetworkSimulation(env2)
random.seed(42) # Same topology
net_sim2.create_topology(num_nodes=15, connectivity=0.2)
# Re-apply bad nodes
for n in net_sim2.nodes: net_sim2.graph.nodes[n]['reliability'] = 0.98
for n in bad_nodes: 
    if n in net_sim2.graph.nodes: net_sim2.graph.nodes[n]['reliability'] = 0.6

net_sim2.simulate_packet = realistic_simulate_packet

env2.process(net_sim2.update_congestion())

trust_model_int = TrustModel()
routing_int = IntelligentRouting(net_sim2.graph, trust_model_int)
proc2 = env2.process(run_simulation(env2, net_sim2, trust_model_int, routing_int, num_packets=50))
env2.run(until=proc2)
```
- **Fresh environment**: Prevents state carryover
- **Same topology**: Uses same seed (42)
- **Expected**: Better performance (learns to avoid bad nodes)

### Lines 182-213: **Run 3: Q-Learning**
```python
print("\n--- Simulation 3: Q-Learning Routing (Training Phase) ---")
env3 = simpy.Environment()
net_sim3 = NetworkSimulation(env3)
random.seed(42)
net_sim3.create_topology(num_nodes=15, connectivity=0.2)
# ... setup bad nodes ...

# Initialize Agent
rl_agent = QLearningAgent(net_sim3.nodes, epsilon=0.5) # High exploration initially
routing_rl = RLRouting(net_sim3.graph, rl_agent)

# Train heavily
proc3 = env3.process(run_simulation(env3, net_sim3, None, routing_rl, num_packets=500))
env3.run(until=proc3)

# Test Phase (Exploit)
print("\n--- Simulation 3b: Q-Learning Routing (Testing Phase) ---")
rl_agent.epsilon = 0.05 # Reduce exploration
env3_test = simpy.Environment()
net_sim3.env = env3_test
env3_test.process(net_sim3.update_congestion())

proc3_test = env3_test.process(run_simulation(env3_test, net_sim3, None, routing_rl, num_packets=50))
env3_test.run(until=proc3_test)
```
- **Two phases**:
  1. Training: 500 packets with ε=0.5 (explore)
  2. Testing: 50 packets with ε=0.05 (exploit)
- **Expected**: High variance initially, improves with learning

### Lines 216-217: Final visualization
```python
visualize_network(net_sim2.graph, trust_model=trust_model_int, filename="network_final_trust.png")
```

---

## File 2: `compare_algos.py` - Benchmark Tool

### Key Function: `run_scenario` (lines 16-126)
Similar to `main.py` but more structured for metrics collection.

### Lines 129-161: Main comparison loop
```python
# 1. Standard
pdr, lat = run_scenario("Standard", ShortestPathRouting)
results['Standard (Dijkstra)'] = {'PDR': pdr, 'Latency': lat}

# 2. Intelligent
pdr, lat = run_scenario("Intelligent", IntelligentRouting)
results['Intelligent (Trust)'] = {'PDR': pdr, 'Latency': lat}

# 3. RL (Train then Test)
agent = QLearningAgent(dummy_net.nodes, epsilon=0.5)
run_scenario("Q-Learning", RLRouting, agent=agent, packets=500, training=True)
agent.epsilon = 0.05
pdr, lat = run_scenario("Q-Learning", RLRouting, agent=agent, packets=100, training=False)
results['Q-Learning (AI)'] = {'PDR': pdr, 'Latency': lat}
```

### Lines 163-198: Plotting
```python
labels = list(results.keys())
pdrs = [results[l]['PDR'] for l in labels]
lats = [results[l]['Latency'] for l in labels]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Dual-axis bar chart
ax1.bar(x - width/2, pdrs, width, color='tab:blue', label='PDR')
ax2 = ax1.twinx()
ax2.bar(x + width/2, lats, width, color='tab:orange', label='Latency')

plt.savefig('comparison_chart.png')
```
- Creates side-by-side comparison chart

---

## File 3: `qos_sim.py` - QoS Testing

### Lines 11-34: Setup
```python
env = simpy.Environment()
net_sim = NetworkSimulation(env)
net_sim.create_topology(num_nodes=15, connectivity=0.3)
env.process(net_sim.update_congestion()) # Key for QoS diff

trust_model = TrustModel()
routing = IntelligentRouting(net_sim.graph, trust_model)

stats = {
    'data': {'sent': 0, 'recv': 0},
    'voice': {'sent': 0, 'recv': 0}
}
```

### Lines 36-60: Traffic generator
```python
def packet_generator():
    for _ in range(200):
        p_type = 'voice' if random.random() < 0.5 else 'data'
        priority = 1 if p_type == 'voice' else 0
        
        src, dst = random.sample(net_sim.nodes, 2)
        path = routing.find_path(src, dst)
        if path:
            success = net_sim.simulate_packet(path, trust_model, priority=priority)
            
            stats[p_type]['sent'] += 1
            if success:
                stats[p_type]['recv'] += 1
        
        yield env.timeout(0.1)
```
- **50/50 mix** of voice and data
- Passes `priority` flag to packet simulation

### Lines 65-75: Results
```python
d_pdr = (stats['data']['recv'] / stats['data']['sent'] * 100)
v_pdr = (stats['voice']['recv'] / stats['voice']['sent'] * 100)

print(f"Data (Low Prio)  | PDR: {d_pdr:.2f}%")
print(f"Voice (High Prio)| PDR: {v_pdr:.2f}%")

if v_pdr > d_pdr:
    print("SUCCESS: Voice traffic had higher reliability during congestion!")
```
- **Expected**: Voice PDR > Data PDR

---

## File 4: `attack_sim.py` - Attack Scenarios

### Lines 12-25: Setup
```python
def run_attack_scenario(attack_type="blackhole"):
    env = simpy.Environment()
    net_sim = NetworkSimulation(env)
    net_sim.create_topology(num_nodes=15, connectivity=0.3)
    
    # Setup Adversaries
    malicious_nodes = [5, 9]
    adversaries = {n: Adversary(n, attack_type) for n in malicious_nodes}
    
    for adv in adversaries.values():
        env.process(adv.update_behavior(env))
```
- Creates 2 malicious nodes
- Starts their behavior processes

### Lines 27-50: Monkey patch with adversary logic
```python
def secure_simulate_packet(path, trust_model, priority=0):
    if not path: return False
    success = True
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        # Check if V is an adversary
        if v in adversaries:
            verdict = adversaries[v].process_packet("data" if priority == 0 else "voice")
            if not verdict:
                logger.debug(f"Packet dropped by ADVERSARY {v} ({attack_type})")
                success = False
                if trust_model: trust_model.update_trust(v, False)
                break
        
        if trust_model: trust_model.update_trust(v, True)
        
    return success

net_sim.simulate_packet = secure_simulate_packet
```
- Intercepts packet simulation
- Calls `adversary.process_packet()` for malicious nodes

### Lines 52-77: Run simulation
```python
trust_model = TrustModel()
routing = IntelligentRouting(net_sim.graph, trust_model)

def traffic():
    for i in range(200):
        src, dst = random.sample(net_sim.nodes, 2)
        if src in malicious_nodes or dst in malicious_nodes:
            continue # Avoid picking adversaries as endpoints
        
        path = routing.find_path(src, dst)
        if path:
            success = net_sim.simulate_packet(path, trust_model)
            stats['total'] += 1
            if success: stats['success'] += 1
        yield env.timeout(0.1)

env.process(traffic())
env.run()

pdr = (stats['success'] / stats['total'] * 100)
print(f"Result for {attack_type}: PDR = {pdr:.2f}%")
```

### Lines 79-83: Test all attacks
```python
if __name__ == "__main__":
    run_attack_scenario("blackhole")
    run_attack_scenario("grayhole")
    run_attack_scenario("on-off")
```
- Runs three separate scenarios
- **Expected results**:
  - Blackhole: Low PDR (~60-70%)
  - Grayhole: Higher PDR (only data packets affected)
  - On-Off: Variable PDR (depends on timing)
