# Tile Placement Intelligence Analysis

## Current State: Agent Cannot Make Intelligent Tile Placement Decisions

### Problem Summary
The agent **does NOT** have access to spatial information needed for intelligent tile placement decisions such as:
- Building bonuses on spaces
- Ocean adjacency (2€ bonus when placing next to ocean)
- Greenery adjacency (city VP when placing next to greenery)
- Spatial relationships between available spaces and existing tiles

### Current Implementation

#### 1. State Encoder (`state_encoder.py`)
**What it encodes:**
- Aggregate board statistics (total cities, greenery, oceans)
- Own tile counts
- City/greenery clustering patterns
- City-greenery combo scoring signals

**What it MISSES:**
- Per-space bonus information (spaces have `bonus` field but it's ignored)
- Adjacency information (which spaces are next to oceans/greenery/cities)
- Spatial features for available spaces when selecting placement

#### 2. Action Decoder (`action_decoder.py`)
**How space selection works:**
- Maps action index (0, 1, 2, ...) to space index
- Simply selects space by position in `availableSpaces` list
- No encoding of which spaces have bonuses or are adjacent to valuable tiles

#### 3. Action Context Encoding (`_encode_action_context`)
**Current encoding for `selectSpace`:**
- Only encodes: `len(spaces) / 40.0` (count of available spaces)
- Does NOT encode: bonuses, adjacency, or any per-space features

### Evidence from Code

**State Encoder - Board State (lines 595-696):**
```python
def _encode_board_state(self, game_state, current_player):
    # Only encodes aggregate counts, not per-space bonuses
    for space in spaces:
        bonus = space.get('bonus', [])  # This exists but is NEVER used!
        # Only counts tiles, doesn't encode bonuses
```

**Action Context (lines 882-884):**
```python
elif input_type in ['selectSpace', 'space']:
    spaces = waiting_for.get('availableSpaces', waiting_for.get('spaces', []))
    encoding[37] = min(len(spaces) / 40.0, 1.0)  # Only count, no features!
```

**Action Decoder (lines 1471-1489):**
```python
elif input_type in ['space', 'selectSpace']:
    spaces = waiting_for.get('availableSpaces') or waiting_for.get('spaces', [])
    idx = normalize_index(action_index, 300)
    # Just selects by index - no intelligence about bonuses/adjacency
    return {'type': 'space', 'spaceId': str(chosen.get('id'))}
```

### What the Agent Can Learn
- Through trial and error, it may learn that certain space indices correlate with better rewards
- However, without explicit spatial features, it cannot generalize intelligently
- It cannot distinguish between spaces with bonuses vs. without
- It cannot prioritize spaces adjacent to oceans/greenery

### Recommended Solutions

#### Option 1: Enhance Action Context Encoding (Recommended)
Add spatial features to `_encode_action_context` when `input_type == 'selectSpace'`:

```python
elif input_type in ['selectSpace', 'space']:
    spaces = waiting_for.get('availableSpaces', waiting_for.get('spaces', []))
    encoding[37] = min(len(spaces) / 40.0, 1.0)
    
    # NEW: Encode spatial features for available spaces
    if spaces and isinstance(spaces[0], dict):
        game_state = player_state.get('game', {})
        all_spaces = game_state.get('spaces', [])
        
        # Aggregate features across available spaces
        total_bonuses = 0
        spaces_with_bonuses = 0
        spaces_adjacent_to_ocean = 0
        spaces_adjacent_to_greenery = 0
        spaces_adjacent_to_city = 0
        
        for space in spaces[:10]:  # Limit to first 10 for encoding
            space_id = space.get('id', '')
            space_bonus = space.get('bonus', [])
            
            if space_bonus:
                total_bonuses += len(space_bonus)
                spaces_with_bonuses += 1
            
            # Calculate adjacency (requires coordinate-based logic)
            # ... adjacency calculation ...
        
        encoding[38] = min(total_bonuses / 20.0, 1.0)
        encoding[39] = min(spaces_with_bonuses / len(spaces), 1.0)
        # ... more features ...
```

#### Option 2: Add Per-Space Features to State
Encode top-K available spaces with their features directly in the state vector.

#### Option 3: Use Attention Mechanism
Implement attention over available spaces, allowing the network to focus on spaces with better features.

### Implementation Priority
1. **High Priority**: Add bonus encoding to action context
2. **High Priority**: Add adjacency calculation and encoding
3. **Medium Priority**: Encode top-K available spaces with full features
4. **Low Priority**: Consider attention mechanism for space selection

### Testing
After implementing, verify:
- Agent learns to prefer spaces with bonuses
- Agent learns to place cities next to greenery
- Agent learns to place tiles next to oceans for 2€ bonus
- Agent makes more strategic placement decisions overall
