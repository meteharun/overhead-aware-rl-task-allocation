"""Learned allocation: environment, policy, training.

The allocator wrapper that plugs the trained policy into the rest of the
project lives in `allocators/learned.py`, registered under the name `rl`, so
nothing imports torch unless a learned allocator is asked for.
"""
