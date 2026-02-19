import sys
import os

# Ensure the models directory is in the path
# Workspace root is c:\Fakturacia\TFM
sys.path.append(os.path.abspath(os.path.join('c:\\Fakturacia\\TFM', 'rl-environment')))

from models.action_decoder import ActionDecoder

def test_wasteful_masking():
    decoder = ActionDecoder()
    
    # State with maxed parameters
    player_state = {
        "game": {
            "temperature": 8,
            "oceans": 9,
            "venusScaleLevel": 30,
            "oxygenLevel": 14
        },
        "thisPlayer": {
            "megaCredits": 100,
            "heat": 8,
            "plants": 8,
            "id": "p1",
            "color": "red"
        },
        "players": [
             {"id": "p1", "color": "red"}
        ],
        "waitingFor": {
            "type": "or",
            "options": [
                {
                    "type": "selectCard",
                    "title": {"message": "Standard Project"},
                    "cards": [
                        {"name": "Asteroid:SP", "isDisabled": False},
                        {"name": "Aquifer", "isDisabled": False},
                        {"name": "Greenery", "isDisabled": False}
                    ]
                },
                {
                    "type": "selectCard",
                    "title": {"message": "Convert Heat"},
                    "cards": []
                }
            ]
        }
    }
    
    available = decoder.get_available_actions(player_state)
    
    # Check that Asteroid:SP (100) and Aquifer (101) are NOT available
    # Check that Greenery (102) IS available
    # Check that Convert Heat (701) is NOT available
    
    print(f"Available actions: {available}")
    
    # Standard Project indices are self.action_types['STANDARD_PROJECT'] + j
    # For j=0 (Asteroid:SP): index 100
    # For j=1 (Aquifer): index 101
    # For j=2 (Greenery): index 102
    
    assert 100 not in available, "Asteroid:SP should be masked (Max Temp)"
    assert 101 not in available, "Aquifer should be masked (Max Oceans)"
    assert 102 in available, "Greenery should NOT be masked (VP Value)"
    assert 701 not in available, "Convert Heat should be masked (Max Temp)"
    
    # State with non-maxed parameters
    player_state_low = {
        "game": {
            "temperature": 0,
            "oceans": 0,
            "venusScaleLevel": 0,
            "oxygenLevel": 0
        },
        "thisPlayer": {
            "megaCredits": 100,
            "heat": 8,
            "plants": 8,
            "id": "p1",
            "color": "red"
        },
        "players": [
             {"id": "p1", "color": "red"}
        ],
        "waitingFor": {
            "type": "or",
            "options": [
                {
                    "type": "selectCard",
                    "title": {"message": "Standard Project"},
                    "cards": [
                        {"name": "Asteroid:SP", "isDisabled": False}
                    ]
                },
                {
                    "type": "selectCard",
                    "title": {"message": "Convert Heat"},
                    "cards": []
                }
            ]
        }
    }
    
    available_low = decoder.get_available_actions(player_state_low)
    print(f"Available low actions: {available_low}")
    
    assert 100 in available_low, "Asteroid:SP should be available (Temp low)"
    assert 701 in available_low, "Convert Heat should be available (Temp low)"
    
    print("Verification successful!")

if __name__ == "__main__":
    test_wasteful_masking()
