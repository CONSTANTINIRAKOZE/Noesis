"""The browser version of Walter (web/walter.js) must match the PyTorch one.

Runs web/walter.js under Node and compares with app/walter_service.py. Skipped if Node isn't installed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.walter_service import Walter

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='Node.js not installed')


@pytest.fixture(scope='module')
def py():
    return Walter()


def js(*calls):
    out = subprocess.run(['node', str(ROOT / 'tests' / 'js_parity.mjs'), json.dumps(calls)],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def close(a, b):
    return {d['char']: d['p'] for d in a} == pytest.approx({d['char']: d['p'] for d in b}, abs=2e-5)


def test_exported_weights_are_current(py):
    meta = json.loads((ROOT / 'web' / 'walter.json').read_text())
    assert meta['n_params'] == py.model.num_params()
    assert meta['dev_loss'] == pytest.approx(py.dev_loss, abs=1e-4), 'run: python -m app.export_weights'


@pytest.mark.parametrize('prefix,ablate', [('', []), ('q', []), ('ann', []), ('constanti', []),
                                           ('ann', [[0, 2], [0, 3]]), ('isabel', [[1, 0], [3, 3]])])
def test_next_letter_matches(py, prefix, ablate):
    [j] = js(['nextLetter', prefix, ablate])
    assert close(j['distribution'], py.next_letter(prefix, [tuple(a) for a in ablate])['distribution'])


def test_score_and_inspect_match(py):
    s, i = js(['score', 'constantin'], ['inspect', 'annabelle', [[0, 2]]])
    ps, pi = py.score('constantin'), py.inspect('annabelle', [(0, 2)])
    assert s['loss_gpt'] == pytest.approx(ps['loss_gpt'], abs=1e-3)
    assert s['loss_bigram'] == pytest.approx(ps['loss_bigram'], abs=1e-3)
    assert i['tokens'] == pi['tokens']
    np.testing.assert_allclose(np.array(i['attention']), np.array(pi['attention']), atol=1e-5)
    for jl, pl in zip(i['lens'], pi['lens']):
        assert close(jl['top'], pl['top'])


def test_generation_and_errors():
    g1, g2, bad = js(['generate', 5, 1.0, 'con', 3], ['generate', 5, 1.0, 'con', 3], ['score', 'x1'])
    assert g1 == g2 and all(n['name'].startswith('con') for n in g1)
    assert 'a-z' in bad['error']
