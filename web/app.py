"""Local CPU inference server. Uploads are temporary and removed after each request."""
import argparse
import os
import tempfile
import cv2
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from scripts.inference import InferenceEngine, WARNING
from scripts.image_analysis import analyze_images, MAX_BYTES
from scripts.image_inference import ImageInferenceEngine

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT=ROOT/'models'/'experimental_cnn.pth'


def create_app(checkpoint=None):
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=64*1024**2
    engine=InferenceEngine(checkpoint or os.environ.get('CRIMESCOPE_CHECKPOINT',DEFAULT_CHECKPOINT))
    app.extensions['inference_engine']=engine
    image_path=ROOT/'models'/'experimental_image_cnn.pth'
    image_engine=None
    if image_path.exists():
        try: image_engine=ImageInferenceEngine(image_path)
        except (ValueError,RuntimeError,KeyError) as error: app.logger.warning('Image model disabled: %s',error)
    app.extensions['image_inference_engine']=image_engine

    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.get('/')
    def index():return render_template('index.html')

    @app.get('/api/model')
    def metadata():return jsonify(engine.metadata)

    @app.get('/api/images/model')
    def image_metadata():
        return jsonify(available=image_engine is not None, model=image_engine.metadata if image_engine else None)

    @app.post('/api/predict')
    def predict():
        upload=request.files.get('video')
        if upload is None or not upload.filename:
            return jsonify(error='Choose a video file.',warning=WARNING),400
        suffix=Path(upload.filename).suffix.lower()
        if suffix not in {'.mp4','.avi','.mov','.mkv','.webm'}:
            return jsonify(error='Supported formats: MP4, AVI, MOV, MKV and WebM.',warning=WARNING),415
        try:
            with tempfile.TemporaryDirectory(prefix='crimescope-upload-') as temporary:
                path=Path(temporary)/('video'+suffix)
                upload.save(path)
                result=engine.predict(path)
            return jsonify(result)
        except (ValueError,RuntimeError,cv2.error) as exc:
            app.logger.info('Video inference rejected: %s',exc)
            return jsonify(error='Unable to decode this video. Try a readable MP4 or AVI file.',warning=WARNING),422

    @app.post('/api/images/analyze')
    def images():
        uploads = request.files.getlist('images')
        if not 1 <= len(uploads) <= 6:
            return jsonify(error='Choose between 1 and 6 images.'),400
        try:
            return jsonify(analyze_images([item.stream.read(MAX_BYTES + 1) for item in uploads],image_engine))
        except ValueError as error:
            return jsonify(error=str(error)),422

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):return jsonify(error='Video exceeds the 64 MB upload limit.',warning=WARNING),413

    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CrimeScope UNVALIDATED experimental video inference')
    parser.add_argument('--checkpoint',default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    import torch
    from waitress import serve
    torch.set_num_threads(2)
    print(f'CrimeScope inference: http://127.0.0.1:{args.port} — {WARNING}',flush=True)
    serve(create_app(args.checkpoint),host='127.0.0.1',port=args.port,threads=2)
