#!/usr/bin/env python3
"""
Test script to debug import issues
"""
import sys
import os

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nCurrent working directory:")
print(f"  {os.getcwd()}")

print("\nFiles in current directory:")
for file in os.listdir('.'):
    print(f"  {file}")

print("\nFiles in models directory:")
if os.path.exists('models'):
    for file in os.listdir('models'):
        print(f"  {file}")
else:
    print("  models directory does not exist")

print("\nTrying to import models.agent...")
try:
    from models.agent import RLAgent
    print("  SUCCESS: models.agent imported successfully")
except ImportError as e:
    print(f"  ERROR: {e}") 