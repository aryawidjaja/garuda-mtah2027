"""TV-feed cue columns for degraded Metrica draws (camera behaviour transplanted from SkillCorner), for
train_final.py."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import experiment_transplant as et  # noqa: E402


def met_cue(mid, seed, lib, X0, X1, per):
    """Cue columns for a degraded Metrica draw (same rows for the plain and swapped views)."""
    blocks = et.cue_blocks(mid, seed, lib)
    B = np.full((len(X0), blocks[0].shape[1]), np.nan, np.float32)
    for k, b in enumerate(blocks):
        m = per == k
        B[m] = b[:m.sum()]
    return np.hstack([X0, B]), np.hstack([X1, B])

