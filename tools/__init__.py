"""
Supporting tools that are not part of the experimental pipeline itself:
provenance capture and the artefact manifest.

Kept separate from `evaluation/` deliberately — nothing in here computes a
scientific result. These modules exist so that every result which IS
computed can be traced back to the exact code, environment and settings
that produced it.
"""
