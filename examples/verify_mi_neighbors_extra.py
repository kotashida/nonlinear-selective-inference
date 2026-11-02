"""Verify three additional MI neighborhoods under fixed combined AIS."""
import sys
from verify_mi_neighbors_sweep import verify_variant

if __name__ == "__main__":
    for neighbors in (35, 40, 50):
        verify_variant(sys.argv[1], neighbors)
