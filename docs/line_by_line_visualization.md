# Visualization & Dashboard - Line-by-Line Explanation

## File 1: `visualization.py` - Network Graph Visualization

### Lines 1-2: Imports
```python
import networkx as nx
import matplotlib.pyplot as plt
```

### Line 4: `def visualize_network(graph, trust_model=None, filename="network_topology.png", return_fig=False):`
- **Parameters**:
  - `graph`: NetworkX graph to visualize
  - `trust_model`: Optional TrustModel for coloring nodes
  - `filename`: Output image path
  - `return_fig`: If True, returns Figure object (for Streamlit)

### Lines 5-9: Docstring
```python
"""
Visualizes the network topology.
- Nodes are colored based on reliability/trust if provided.
- Bad/Low Trust nodes -> Red
- Good/High Trust nodes -> Green
"""
```

### Line 11: `fig = plt.figure(figsize=(10, 8))`
- Creates 10x8 inch figure

### Line 13: `pos = nx.spring_layout(graph, seed=42)`
- **Spring layout**: Force-directed algorithm
- **seed=42**: Ensures consistent positioning across runs
- Simulates physical system where edges are springs

### Line 15: `node_colors = []`
- List to store colors for each node

### Lines 16-25: **Coloring with Trust Model**
```python
if trust_model:
    # If we have a trust model, use its values
    for node in graph.nodes():
        trust = trust_model.get_trust(node)
        # Interpolate Red (0) to Green (1)
        # Simple threshold for now
        if trust < 0.5:
            node_colors.append('red')
        else:
            node_colors.append('green')
```
- **Red**: Untrusted nodes (trust < 0.5)
- **Green**: Trusted nodes (trust ≥ 0.5)

### Lines 26-33: **Coloring with Reliability Attribute**
```python
else:
    # Fallback to checking 'reliability' attribute
    for node in graph.nodes():
        rel = graph.nodes[node].get('reliability', 1.0)
        if rel < 0.8:
            node_colors.append('red')
        else:
            node_colors.append('green')
```
- Uses ground truth reliability if no trust model

### Line 35-36: Draw nodes
```python
# Draw nodes
nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)
```
- **node_size=500**: Medium-sized circles

### Lines 38-39: Draw edges
```python
# Draw edges
nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.5)
```
- **arrows=True**: Shows direction (for directed graph)
- **alpha=0.5**: 50% transparency

### Lines 41-42: Draw labels
```python
# Labels
nx.draw_networkx_labels(graph, pos)
```
- Shows node IDs

### Lines 44-46: Draw edge labels
```python
# Edge Labels (Weights/Latency)
edge_labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
```
- Shows latency values on edges

### Line 48: `plt.title("Network Topology (Green=Trusted, Red=Untrusted)")`

### Lines 50-52: Return figure (for Streamlit)
```python
if return_fig:
    plt.close(fig) # Close global pyplot ref but return object
    return fig
```
- Prevents duplicate display in Streamlit

### Lines 54-60: Save to file
```python
try:
    plt.savefig(filename)
    print(f"Network visualization saved to {filename}")
except Exception as e:
    print(f"Error saving visualization: {e}")
finally:
    plt.close()
```
- Error handling and cleanup

---

## File 2: `dashboard.py` - Streamlit Web Interface

This is the largest file (381 lines). I'll focus on key sections.

### Lines 1-13: Imports
```python
import streamlit as st
import simpy
import networkx as nx
import random
import matplotlib.pyplot as plt
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import IntelligentRouting, ShortestPathRouting, RLRouting, RIPRouting
from rl_agent import QLearningAgent
from visualization import visualize_network
import pandas as pd
from advanced_agents import DQNAgent, GNNRLAgent
```

### Lines 15-38: **Session State Initialization**
```python
if 'env' not in st.session_state:
    st.session_state.env = simpy.Environment()
    st.session_state.net_sim = NetworkSimulation(st.session_state.env)
    st.session_state.net_sim.create_topology(num_nodes=15, connectivity=0.2)
    
    # Pre-seed some bad nodes for demo
    st.session_state.net_sim.graph.nodes[3]['reliability'] = 0.5
    st.session_state.net_sim.graph.nodes[7]['reliability'] = 0.6
    
    st.session_state.trust_model = TrustModel()
    
    # Initialize Agents
    nodes = list(st.session_state.net_sim.graph.nodes)
    st.session_state.rl_agent = QLearningAgent(nodes)
    st.session_state.dqn_agent = DQNAgent(nodes)
    st.session_state.gnn_agent = GNNRLAgent(st.session_state.net_sim.graph, nodes)
    
    # Default Routing
    st.session_state.routing_algo_name = "Intelligent (Trust)"
    st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    
    st.session_state.packet_stats = []
    st.session_state.time = 0
```
- **`st.session_state`**: Persists data across Streamlit reruns
- **Only runs once**: `if 'env' not in...` prevents reinitialization
- Creates all agents upfront

### Lines 40-66: **Sidebar Algorithm Selection**
```python
st.sidebar.header("Simulation Controls")

algo_option = st.sidebar.selectbox(
    "Routing Protocol", 
    ["Standard OSPF (Latency)", "RIP (Hop Count)", "Intelligent (Trust)", 
     "Q-Learning (AI)", "DQN (Deep RL)", "GNN-RL (Graph AI)"]
)

# Handle Algorithm Change
if algo_option != st.session_state.routing_algo_name:
    st.session_state.routing_algo_name = algo_option
    if algo_option == "Standard OSPF (Latency)":
        st.session_state.routing = ShortestPathRouting(st.session_state.net_sim.graph)
    elif algo_option == "RIP (Hop Count)":
        st.session_state.routing = RIPRouting(st.session_state.net_sim.graph)
    elif algo_option == "Intelligent (Trust)":
        st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    # ... (similar for RL variants)
    
    st.toast(f"Switched to {algo_option}")
```
- **Dropdown**: User selects routing algorithm
- **Dynamic switching**: Updates `st.session_state.routing`
- **Toast notification**: Confirms switch

### Lines 68-76: **Node Reliability Controls**
```python
st.sidebar.subheader("Node Reliability")
nodes = list(st.session_state.net_sim.graph.nodes)
selected_node = st.sidebar.selectbox("Select Node to Modify", nodes)
reliability = st.sidebar.slider(f"Reliability of Node {selected_node}", 0.0, 1.0, 
                                st.session_state.net_sim.graph.nodes[selected_node].get('reliability', 1.0))

if st.sidebar.button("Update Node"):
    st.session_state.net_sim.graph.nodes[selected_node]['reliability'] = reliability
    st.success(f"Node {selected_node} reliability set to {reliability}")
```
- Allows user to manually degrade nodes

### Lines 112-136: **Traffic Schedule Management**
```python
if 'traffic_schedule' not in st.session_state:
    st.session_state.traffic_schedule = []

def get_traffic_pair(index, nodes, graph):
    """
    Returns the (src, dst) pair for the given index.
    Generates and caches it if it doesn't exist.
    """
    while len(st.session_state.traffic_schedule) <= index:
        attempts = 0
        while True:
            src, dst = random.sample(nodes, 2)
            if nx.has_path(graph, src, dst):
                st.session_state.traffic_schedule.append((src, dst))
                break
            attempts += 1
            if attempts > 20: 
                # Fallback: use largest connected component
                largest_cc = max(nx.weakly_connected_components(graph), key=len)
                src, dst = random.sample(list(largest_cc), 2)
                st.session_state.traffic_schedule.append((src, dst))
                break
    
    return st.session_state.traffic_schedule[index]
```
- **Deterministic traffic**: Same source-dest pairs for fair comparison
- **Infinite schedule**: Generates on demand

### Lines 138-148: **Simulation Controls**
```python
st.sidebar.subheader("Run Simulation")
cols = st.sidebar.columns(2)
train_mode = cols[0].button("🚀 Train Agent")
run_mode = cols[1].button("▶️ Run Step")
steps = st.sidebar.number_input("Packets", min_value=1, value=50)

if st.sidebar.button("🔄 Regenerate Traffic"):
    st.session_state.traffic_schedule = []
    st.success("Traffic schedule cleared.")
```
- **Train**: High exploration for learning
- **Run Step**: Low exploration for testing

### Lines 149-277: **Main Simulation Function**
```python
def run_simulation_batch(num_packets, training=False):
    env = st.session_state.env
    net_sim = st.session_state.net_sim
    routing = st.session_state.routing
    trust_model = st.session_state.trust_model
    
    # Validation: Training only for RL
    if training and not hasattr(routing, 'agent'):
        st.warning("Training is only available for AI/RL algorithms.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    rewards = []
    
    # Epsilon Decay
    start_epsilon = routing.agent.epsilon if hasattr(routing, 'agent') else 0.1
    end_epsilon = 0.01
    decay_rate = 0.995
```
- **Progress bar**: Visual feedback
- **Epsilon decay**: Exploration → Exploitation

### Lines 194-253: **Packet Processing Loop**
```python
for i in range(num_packets):
    idx_to_use = i if not training else (start_idx + i)
    src, dst = get_traffic_pair(idx_to_use, nodes, net_sim.graph)

    # Epsilon Decay
    if training and hasattr(routing, 'agent'):
        routing.agent.epsilon = max(end_epsilon, routing.agent.epsilon * decay_rate)
    
    path = routing.find_path(src, dst)
    success = True
    
    if path:
        # Simulate Traversal
        for j in range(len(path) - 1):
             u, v = path[j], path[j+1]
             rel = net_sim.graph.nodes[v].get('reliability', 1.0)
             if random.random() > rel:
                 success = False
                 trust_model.update_trust(v, False)
                 break
             else:
                 trust_model.update_trust(v, True)
        
        # Feedback / Learning
        if hasattr(routing, 'agent'):
            reward = (10 - 0.1 * len(path)) if success else -10
            rewards.append(reward)
            
            # Learn
            for j in range(len(path) - 1):
                u_node = path[j]
                v_node = path[j+1]
                next_neighbors = list(net_sim.graph.neighbors(v_node))
                
                routing.agent.learn(u_node, v_node, reward, v_node, next_neighbors, target_node=dst)
```
- **Reward shaping**: `10 - 0.1*hops` (encourages short paths)

### Lines 283-318: **Live Simulation Tab**
```python
tab1, tab2 = st.tabs(["🔴 Live Simulation", "📊 Benchmark Comparison"])

with tab1:
    view_mode = st.radio("Visualization Mode", 
                         ["Ground Truth (Reliability)", "Agent Perception (Trust Score)"])

    with col1:
        st.subheader(f"Network Topology ({view_mode})")
        
        tm_to_use = st.session_state.trust_model if view_mode == "Agent Perception (Trust Score)" else None
        
        fig = visualize_network(st.session_state.net_sim.graph, 
                                trust_model=tm_to_use, 
                                return_fig=True)
        st.pyplot(fig)
```
- **Two views**: Ground truth vs. learned trust

### Lines 302-317: **Statistics Display**
```python
with col2:
    st.subheader("Recent Traffic")
    if st.session_state.packet_stats:
        df = pd.DataFrame(st.session_state.packet_stats[-10:]) # Last 10
        st.dataframe(df)
        
        # Stats
        all_df = pd.DataFrame(st.session_state.packet_stats)
        pdr = (all_df[all_df['Status'] == 'Success'].shape[0] / all_df.shape[0]) * 100
        st.metric("Packet Delivery Ratio", f"{pdr:.2f}%")
    else:
        st.info("Run simulation to see stats.")

# Trust Table
st.subheader("Node Trust Scores")
trust_data = {n: st.session_state.trust_model.get_trust(n) for n in nodes}
st.bar_chart(trust_data)
```
- Shows last 10 packets
- Bar chart of trust scores

### Lines 319-380: **Benchmark Tab**
```python
with tab2:
    st.subheader("🏆 Protocol Benchmark")
    st.markdown("Compare different algorithms on the **exact same traffic**.")
    
    comp_algos = st.multiselect("Select Algorithms to Compare", 
                                ["Shortest Path", "Intelligent Routing", "Q-Learning", 
                                 "DQN (Deep RL)", "GNN-RL (Graph AI)"],
                                default=["Shortest Path", "DQN (Deep RL)"])
    
    comp_steps = st.number_input("Comparison Packets", min_value=10, value=100)
    
    if st.button("Run Benchmark"):
        results = []
        my_bar = st.progress(0)
        
        chart_data = {'Packet': [], 'Algorithm': [], 'Cumulative PDR': []}
        
        for algo_name in comp_algos:
            bench_routing = get_routing_algo(algo_name, ...)
            successes = 0
            
            for i in range(comp_steps):
                src, dst = get_traffic_pair(i, ...)  # Same traffic!
                path = bench_routing.find_path(src, dst)
                
                # ... simulate packet ...
                
                chart_data['Packet'].append(i + 1)
                chart_data['Algorithm'].append(algo_name)
                current_pdr = (successes / (i + 1)) * 100
                chart_data['Cumulative PDR'].append(current_pdr)
        
        # Plot results
        chart_df = pd.DataFrame(chart_data)
        st.line_chart(chart_df, x="Packet", y="Cumulative PDR", color="Algorithm")
```
- **Apples-to-apples comparison**: Same traffic schedule
- **Real-time chart**: PDR evolves as packets are sent

---

## Key Streamlit Concepts

### Session State
- **Purpose**: Persist data across reruns
- **Usage**: `st.session_state.my_var = value`
- **Critical**: Without it, agents would reset every interaction

### Reactive UI
- Every button click triggers full script re-execution
- State preservation prevents data loss

### Progress Feedback
- `st.progress()`: Progress bar
- `st.empty()`: Placeholder for dynamic updates
- Updates visible during long-running loops
