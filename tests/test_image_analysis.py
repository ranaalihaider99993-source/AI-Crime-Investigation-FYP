import io
import hashlib
import pytest
from PIL import Image
from scripts.image_analysis import analyze_images, MAX_BYTES


def picture():
    stream = io.BytesIO()
    Image.new('RGB', (40, 30), 'gray').save(stream, format='PNG')
    return stream.getvalue()


def test_grounded_multiple_images():
    data = picture()
    result = analyze_images([data, data])
    assert result['semantic_model_available'] is False
    assert len(result['images']) == 2
    assert result['images'][0]['sha256'] == hashlib.sha256(data).hexdigest()
    assert result['images'][0]['width'] == 40
    assert result['scenario']['hypotheses'] == ['No scene or crime hypothesis is supported by this technical inspection.']
    assert all('decoded still image' in text for text in result['scenario']['observations'])


@pytest.mark.parametrize('payload', [[], [b'bad'], [b'x' * (MAX_BYTES + 1)], [picture()] * 7])
def test_reject_invalid(payload):
    with pytest.raises(ValueError):
        analyze_images(payload)


def test_pixel_limit():
    stream = io.BytesIO()
    Image.new('1', (4001, 4000)).save(stream, format='PNG')
    with pytest.raises(ValueError, match='16 million'):
        analyze_images([stream.getvalue()])


def test_image_endpoint(monkeypatch):
    import web.app as module
    class Stub:
        def __init__(self, *args): pass
    monkeypatch.setattr(module, 'InferenceEngine', Stub)
    client = module.create_app().test_client()
    response = client.post('/api/images/analyze', data={'images': [(io.BytesIO(picture()), 'a.png'), (io.BytesIO(picture()), 'b.png')]})
    assert response.status_code == 200
    assert len(response.json['images']) == 2
    assert client.post('/api/images/analyze').status_code == 400
    assert client.post('/api/images/analyze', data={'images': (io.BytesIO(b'bad'), 'a.png')}).status_code == 422
    assert b'Upload video' in client.get('/').data
