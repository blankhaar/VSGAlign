"""Small I/O helpers for derivation and metadata artifacts."""

import json
from pathlib import Path
from typing import Dict, Any

def load_derivation(path: str) -> Dict[str, Any]:
    """Load a JSON derivation artifact used during validation."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Derivation file not found: {path}")
    
    with open(p, 'r') as f:
        data = json.load(f)
        
    return data
