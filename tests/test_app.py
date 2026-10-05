"""API tests for Walter's web app. Run from the repository root: python -m pytest tests"""
import pytest
from fastapi.testclient import TestClient

from app.server import app


@pytest.fixture(scope='module')
def client():
    return TestClient(app)


def test_info(client):
    info = client.get('/api/info').json()
    assert info['n_layer'] == 4 and info['n_head'] == 4
    assert info['params'] > 100_000
    assert info['dev_loss'] < 2.1          # Walter beats the Rung 3 MLP


def test_frontend_is_served(client):
    page = client.get('/')
    assert page.status_code == 200 and '<title>Walter</title>' in page.text
    assert client.get('/static/app.js').status_code == 200


def test_generate_respects_prefix_and_seed(client):
    body = {'n': 8, 'prefix': 'con', 'seed': 7}
    first = client.post('/api/generate', json=body).json()['names']
    again = client.post('/api/generate', json=body).json()['names']
    assert len(first) == 8 and first == again
    assert all(n['name'].startswith('con') for n in first)
    assert all(set(n['name']) <= set('abcdefghijklmnopqrstuvwxyz') for n in first)


def test_next_letter_is_a_distribution(client):
    dist = client.post('/api/next', json={'prefix': 'q'}).json()['distribution']
    assert len(dist) == 27
    assert abs(sum(d['p'] for d in dist) - 1) < 1e-3
    assert dist[0]['char'] == 'u'           # after q comes u


def test_score_compares_gpt_and_bigram(client):
    s = client.post('/api/score', json={'name': 'Emma'}).json()
    assert s['name'] == 'emma' and len(s['steps']) == 5   # e m m a + end
    assert s['steps'][-1]['char'] == '.'
    assert s['loss_gpt'] < s['loss_bigram']
    assert s['in_dataset']


def test_inspect_shapes(client):
    r = client.post('/api/inspect', json={'name': 'anna'}).json()
    assert r['tokens'] == ['.', 'a', 'n', 'n', 'a']
    assert len(r['attention']) == 4 and len(r['attention'][0]) == 4
    row = r['attention'][0][0][2]
    assert len(row) == 5 and abs(sum(row) - 1) < 1e-4 and row[3] == row[4] == 0   # causal mask
    assert len(r['lens']) == 5


def test_ablating_previous_letter_heads_breaks_no_triple_rule(client):
    """Rung 6's finding, through the API: without L0 H2/H3 Walter forgets letters don't triple."""
    p_n = lambda body: next(d['p'] for d in client.post('/api/next', json=body).json()['distribution'] if d['char'] == 'n')
    intact = p_n({'prefix': 'ann'})
    ablated = p_n({'prefix': 'ann', 'ablate': [[0, 2], [0, 3]]})
    assert ablated > 3 * intact
    # ablation must not leak into later requests
    assert p_n({'prefix': 'ann'}) == intact


@pytest.mark.parametrize('path,body', [
    ('/api/score', {'name': ''}),
    ('/api/score', {'name': 'abc1'}),
    ('/api/score', {'name': 'a' * 16}),
    ('/api/next', {'prefix': 'a' * 15}),
    ('/api/inspect', {'name': 'ann', 'ablate': [[4, 0]]}),
    ('/api/generate', {'n': 0}),
    ('/api/generate', {'temperature': 9}),
])
def test_bad_input_is_rejected(client, path, body):
    assert client.post(path, json=body).status_code == 422
