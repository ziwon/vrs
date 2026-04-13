"""VRS — Video Reasoning System.

Two-stage cascade:
  1. YOLOE-L open-vocabulary detector (fast path, ~6 ms / frame).
  2. Cosmos-Reason2-2B physical-reasoning VLM (slow path, runs only on candidates).
"""

__version__ = "0.2.0"
