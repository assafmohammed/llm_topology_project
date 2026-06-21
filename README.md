# LLM Topology Project

This project compares datacenter topologies for LLM-like training traffic.

Current first milestone:
- Build HyperX for 32 GPUs.
- Preserve 8-GPU HBI domains.
- Generate synthetic TP/DP/PP traffic.
- Compute shortest-hop distances for active communication pairs.

Run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python experiments/run_hyperx32.py
pytest -q