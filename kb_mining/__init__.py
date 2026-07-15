"""Kaggle solution mining for KB updates.

The one-shot mining pipeline is harvest -> extract -> aggregate -> decide.
Each stage is idempotent and can be rerun independently; stages communicate only
through files under kb_mining/data/. See docs/kb_mining_protocol.md.
"""
