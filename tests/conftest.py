"""Gemeinsame Fixtures: der Grundfall und die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sre_constants as C  # noqa: E402
import sre_grid as G  # noqa: E402
from sre_model import Par  # noqa: E402
from sre_scenario import DEFAULTS, par_from_settings  # noqa: E402


@pytest.fixture(scope="session")
def grundfall():
    """Modellparameter des Grundfalls (die Voreinstellungen der App)."""
    return par_from_settings(DEFAULTS)


@pytest.fixture(scope="session")
def pre():
    """Die vorgerechneten Regeltabellen aller Presets (data/regeln.csv)."""
    out = G.load_regeln()
    assert out, "data/regeln.csv fehlt: python sre_grid.py regeln"
    return out


@pytest.fixture(scope="session")
def raster():
    out = G.load_raster()
    assert out, "data/raster.csv fehlt: python sre_grid.py raster"
    return out


def mini(**kw) -> Par:
    """Kleine Instanz: 8 Wochen, 2 Jahre, sonst die Voreinstellungen (für Aufzählung und Handrechnung)."""
    return Par(**{**dict(W=8, Y=2), **kw})


@pytest.fixture(scope="module")
def fast_limits():
    """Für App-Tests: weniger Trainingspfade der Schnellabstimmung und kürzeres Profil, damit ein Lauf Sekunden dauert. Modul-Umfang: test_claims.py rechnet mit den echten Werten."""
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "P_TRAIN_QUICK", 12)
    mp.setattr(C, "LIVE_PROFILE_PATHS", 50)
    yield
    mp.undo()
