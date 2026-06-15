"""idm.definitions — the named-config DSL (objects, cuts, hists).

Following the SIDM pattern, the analysis content is expressed as dicts of *named*
callables. A YAML config (see ``idm/configs/``) then selects objects / cuts / hists
*by name*, so studies vary configuration instead of forking code. These modules are
additive scaffolding; ``python_analysis/`` remains the source of truth until ported.
"""
