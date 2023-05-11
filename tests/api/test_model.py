import pytest


def test_predict(client):
    response = client.post('/api/v1/model/predict', json={})
    assert response.status_code == 200
    assert response.json()['prediction'] == 'fake prediction'
