"""gridmon: grid-event monitor for BESS PCS voltage records (simulation-only reference).

The Python package is the reference implementation for a later portable C99 float32 core
(TI C2000 and WebAssembly). Not validated on measured data; not for protection or control.
"""

__version__ = "0.2.0"
GENERATOR_VERSION = "0.2.0"

CLASSES: tuple[str, ...] = (
    "normal",
    "sag_bal",
    "sag_unbal",
    "swell",
    "phase_jump",
    "freq_event",
    "oscillation",
)
