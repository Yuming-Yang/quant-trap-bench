"""Task generators, one module per trap.

Each module exposes ``build(task_id, seed, control, params)``, which returns a
:class:`qtb.traps.base.Built`, and two solvers that read only the public files:
``naive`` (the textbook analysis on the data as given) and ``oracle`` (the reference
correction). The solvers are an independent second implementation: the key is computed
from the generator's internal arrays, the solvers from the CSV files an agent sees.
"""
