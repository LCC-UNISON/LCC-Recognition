

from flask import Flask,render_template,Response,json,jsonify
import cv2
from test import LCCRecognition


app=Flask(__name__)
@app.before_first_request
def do_something_only_once():
    global recognition 
    recognition = LCCRecognition(model='./model/20170512-110547.pb',id_folder=['./ids/'],umbral=1.15)

    
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    #print(Response(recognition.Visualizar(),mimetype='multipart/x-mixed-replace; boundary=frame'))
    return Response(recognition.Visualizar(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)
    
