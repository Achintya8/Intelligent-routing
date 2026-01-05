# Security & Attacks - Line-by-Line Explanation

## File: `security.py`

### Lines 1-2: Imports
```python
import random
import logging
```

### Line 4: `logger = logging.getLogger("Security")`
- Creates logger specific to security module

### Line 6: `class Adversary:`
- Simulates malicious network nodes
- Implements three attack patterns

### Line 7: `def __init__(self, node_id, attack_type="blackhole"):`
- **Parameters**:
  - `node_id`: ID of the compromised node
  - `attack_type`: One of `"blackhole"`, `"grayhole"`, `"on-off"`

### Lines 8-11: Instance variables
```python
self.node_id = node_id
self.attack_type = attack_type
self.active = True
self.state = "good" # For On-Off attacks
```
- `active`: Enables/disables attack
- `state`: Tracks current behavior in on-off attack

---

### Line 13: `def process_packet(self, packet_type="data"):`
- Decides whether to forward or drop a packet
- **Parameter `packet_type`**: `"data"` or `"voice"`
- **Returns**: `True` (forward) or `False` (drop)

### Lines 14-16: Docstring
```python
"""
Returns True if packet is forwarded, False if dropped.
"""
```

### Lines 17-18: Check if attack is active
```python
if not self.active:
    return True # Behaves normally if attack is inactive
```
- Allows disabling attack without removing adversary

---

### Lines 20-21: **Blackhole Attack**
```python
if self.attack_type == "blackhole":
    return False # Drop everything
```
- **Most aggressive**: Drops 100% of packets
- **Easy to detect**: Trust score rapidly drops to 0
- **Impact**: Causes routing failures if node is on critical path

---

### Lines 23-28: **Grayhole Attack**
```python
elif self.attack_type == "grayhole":
    # Selectively drop 'data' packets, forward 'voice' to stay under radar
    # Or drop specific % to annoy but not trigger instant ban
    if packet_type == "data":
        return False
    return True
```
- **Selective dropping**: Only drops low-priority packets
- **Stealth**: Forwards voice packets to maintain some trust
- **Rationale**: Harder to detect than blackhole
- **Impact**: Degrades QoS without complete failure

---

### Lines 30-33: **On-Off Attack**
```python
elif self.attack_type == "on-off":
    if self.state == "bad":
        return False
    return True
```
- **Dynamic behavior**: Alternates between good and malicious
- **Reads state**: `self.state` controlled by `update_behavior()`
- **Stealth**: Rebuilds trust during "good" phases

---

### Line 35: `return True`
- Default: Forward packet if attack type is unknown

---

### Line 37: `def update_behavior(self, env):`
- **SimPy process** that changes attack state over time
- **Parameter `env`**: SimPy environment for timing

### Lines 38-39: Infinite loop
```python
"""For dynamic attacks (On-Off)"""
while True:
```

### Lines 40-49: On-Off cycle logic
```python
if self.attack_type == "on-off":
    # Behave good for 20s to build trust
    self.state = "good"
    logger.debug(f"Adversary {self.node_id}: Switching to GOOD state")
    yield env.timeout(20)
    
    # Attack for 5s
    self.state = "bad"
    logger.debug(f"Adversary {self.node_id}: Switching to BAD state")
    yield env.timeout(5)
```
- **Good phase**: 20 time units (builds trust via successful forwarding)
- **Bad phase**: 5 time units (drops packets)
- **Cycle**: Repeats indefinitely (20s good, 5s bad, 20s good...)
- **Challenge for trust model**: Must detect and respond to this pattern

### Lines 50-51: Static attack handling
```python
else:
    yield env.timeout(100) # Static attacks don't change
```
- Blackhole/Grayhole don't need state changes
- Long timeout prevents busy-waiting

---

## Attack Patterns Summary

### 1. Blackhole Attack
- **Behavior**: Drops 100% of packets
- **Detection**: Very fast (few failures)
- **Countermeasure**: Trust model quickly isolates node
- **Real-world example**: Compromised router, hardware failure

### 2. Grayhole Attack
- **Behavior**: Selectively drops packets by type
- **Detection**: Moderate speed (depends on traffic mix)
- **Countermeasure**: QoS-aware trust tracking
- **Real-world example**: Congestion attack, resource exhaustion

### 3. On-Off Attack
- **Behavior**: Alternates between normal and malicious
- **Detection**: Slow (trust oscillates)
- **Countermeasure**: Requires long-term monitoring, slower trust recovery
- **Real-world example**: Intermittent failures, strategic attack

---

## Trust Model Response

Given the trust update formula from `trust_model.py`:
- **Success**: `trust += 0.05` (capped at 1.0)
- **Failure**: `trust *= 0.95`

### Blackhole Impact:
- After 14 dropped packets: trust ≈ 0.5
- After 28 dropped packets: trust ≈ 0.25
- **Quickly isolated** by Intelligent Routing

### On-Off Impact:
```
Good phase (20 successes): 0.5 → 1.0 (full recovery)
Bad phase (5 failures):    1.0 → 0.77
```
- Node maintains relatively high trust
- **Harder to isolate**, remains in routing paths
- Requires tuning: Lower `bonus_factor` or higher `decay_factor`
