import os
from flask import Flask,url_for,session,redirect,flash
from flask import render_template
from flask import request
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index Page!'
@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
#    return 'Hello World'
    return render_template('hello.html',name=name)
@app.route('/user/<username>')
def show_user_profile(username):
    return 'User %s ' % username
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return 'Post %d' % post_id
@app.route('/projects/')
def projects():
    return 'The project page'
@app.route('/about')
def about():
    return 'The about page'
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/'+ secure_filename(f.filename))

def check_login():
    if 'username' in session:
        return True
    else:
        return False
@app.route('/login', methods=['POST', 'GET'])
def login():
    if check_login():
        return redirect(url_for('stats',uname=session['username']))
    else:
        if request.method == 'GET':
            return render_template('login.html')
        else:
            username=request.form['username'].strip()
            pwd=request.form['pwd'].strip()
            if username=='test' and pwd=='mytest':
                session['username']=username
                return redirect(url_for('stats',uname=username))
            else:
                return 'ERROR!!!'

@app.route('/position')
def stats():
    uname=session['username']#request.args.get('uname')
#    return 'username: %s ' % uname
    return render_template('position.html',name=uname)
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))
@app.route('/reid')
def reid():
    uname=session['username']#request.args.get('uname')
    return render_template('reid.html',name=uname)
@app.route('/trace')
def trace():
    uname=session['username']#request.args.get('uname')
    return render_template('trace.html',name=uname)
@app.route('/startposition')
def startposition():
    cmd="nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /home/kcloud/Mask_RCNN-master/:/host waleedka/modern-deep-learning:mask1.0 jupyter notebook --allow-root /host"
    r=os.popen(cmd)
    text=r.read()
    uname=session['username']
    return render_template('position.html',name=uname)
@app.route('/stopposition')
def stopposition():
    res1=os.popen("docker ps |grep waleedka").read()
    dockername=res1.split()[0]
    print "dockername : "+dockername
    cmd="docker stop "+dockername
    r=os.popen(cmd)
    text=r.read()
    uname=session['username']
    return render_template('position.html',name=uname)
@app.route('/maskrcnn')
def maskrcnn():
    return redirect("http://219.245.186.42:8888/tree/samples")
@app.route('/startreid')
def startreid():
    cmd="nvidia-docker run -it -p 6006:6006 -v /home/kcloud/person-reid-master/:/notebooks/reid -v /home/kcloud/person-reid-master/tmppersons_crop/:/notebooks/persons tensorflow/tf-server-reid:v1.1"
    r=os.popen(cmd)
    text=r.read()
    uname=session['username']
    return render_template('reid.html',name=uname)
@app.route('/stopreid')
def stopreid():
    res1=os.popen("docker ps |grep reid").read()
    dockername=res1.split()[0]
    print "dockername : "+dockername
    cmd="docker stop "+dockername
    r=os.popen(cmd)
    text=r.read()
    uname=session['username']
    return render_template('reid.html',name=uname)
@app.route('/reidaction')
def reidaction():
    uname=session['username']
    res1=os.popen("docker ps |grep reid").read()
    dockername=res1.split()[0]
    print "dockername : "+dockername
    cmd='docker exec -it '+dockername+' bash -c "cd reid && python run_same3.py --mode=test"'
    r=os.popen(cmd)
    text=r.read()
    return render_template('reid.html',name=uname,isok=True)
@app.route('/traceaction')
def traceaction():
    uname=session['username']
    return redirect("http://219.245.186.42:8888/tree/samples/drawline.ipynb")
@app.route('/traceresult')
def traceresult():
    uname=session['username']
    return render_template('trace.html',name=uname,image='static/images/p1.jpg')
if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(host='0.0.0.0',debug=True)
