"""Local CPU inference server. Uploads are temporary and removed after each request."""
import argparse
import os
import tempfile
import hashlib
import threading
import cv2
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from scripts.inference import InferenceEngine, WARNING
from scripts.image_analysis import analyze_images, MAX_BYTES
from scripts.image_inference import ImageInferenceEngine
from scripts.image_analysis import inspect_image
from scripts.incident_analysis import aggregate

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
    incident_lock=threading.Lock()

    @app.after_request
    def headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.get('/')
    def index():return render_template('incident.html')

    @app.get('/individual')
    def individual():return render_template('index.html')

    @app.post('/api/incidents/analyze')
    def incident():
        uploads=request.files.getlist('evidence')
        if not 1 <= len(uploads) <= 9:
            return jsonify(error='Choose 1–6 images and/or 1–3 videos, up to 9 files.'),400
        kinds=[]
        for upload in uploads:
            suffix=Path(upload.filename or '').suffix.lower()
            if suffix in {'.png','.jpg','.jpeg','.webp'}: kinds.append('image')
            elif suffix in {'.mp4','.avi','.mov','.mkv','.webm'}: kinds.append('video')
            else: return jsonify(error='Unsupported evidence format. Use JPEG, PNG, WebP or a supported video.'),415
        if kinds.count('image')>6 or kinds.count('video')>3:
            return jsonify(error='Maximum 6 images and 3 videos per incident.'),400
        if 'image' in kinds and image_engine is None:
            return jsonify(error='The verified image checkpoint is unavailable. No combined result was produced.'),503
        if not incident_lock.acquire(blocking=False):
            return jsonify(error='An incident is already being analyzed. Retry when it finishes.'),429
        try:
            records=[]; total=0; cached={}
            with tempfile.TemporaryDirectory(prefix='crimescope-incident-') as folder:
                for index,(upload,kind) in enumerate(zip(uploads,kinds),1):
                    path=Path(folder)/(str(index)+Path(upload.filename).suffix.lower())
                    digest=hashlib.sha256(); size=0
                    with path.open('xb') as target:
                        while chunk:=upload.stream.read(1024**2):
                            total+=len(chunk); size+=len(chunk)
                            if total>60*1024**2 or (kind=='image' and size>MAX_BYTES):
                                return jsonify(error='Limit: 60 MB total; 8 MB per image.'),413
                            digest.update(chunk); target.write(chunk)
                    key=(kind,digest.hexdigest())
                    record=dict(id=f'file-{index}',name=upload.filename,kind=kind,sha256=key[1],bytes=size)
                    if key in cached:
                        record.update(scores=cached[key]['scores'],duplicate_of=cached[key]['id'])
                    elif kind=='image':
                        data=path.read_bytes(); details=inspect_image(data,index)
                        result=image_engine.predict(data)
                        record.update(scores=result['scores'],model=result['model'],
                                      details={k:v for k,v in details.items() if k!='preview'})
                    else:
                        result=engine.predict(path)
                        record.update(scores=result['scores'],model=result['model'],details=result['preprocessing'])
                    cached.setdefault(key,record); records.append(record)
            combined=aggregate(records)
            combined.update(files=records,warning=WARNING)
            return jsonify(combined)
        except (ValueError,RuntimeError,cv2.error) as error:
            app.logger.info('Incident rejected: %s',error)
            return jsonify(error='Unable to analyze all evidence. Check for corrupt or unreadable files. No combined result was produced.'),422
        finally: incident_lock.release()

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
    def too_large(error):return jsonify(error='Upload exceeds the 64 MB request limit. Incident evidence must total at most 60 MB.',warning=WARNING),413

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
