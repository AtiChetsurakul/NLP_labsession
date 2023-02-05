from flask import Flask, render_template, redirect, url_for, flash, request, send_from_directory
from flask_bootstrap import Bootstrap
from flask_ckeditor import CKEditor
from flask_login import UserMixin, login_user, LoginManager, login_required, current_user, logout_user
from form_ import *
# from flask_sqlalchemy import SQLAlchemy
from flask_wtf.file import FileField
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import os
from functools import wraps
import pyperclip3 as pyclip
import datetime as dt


PASSWORD_STR = os.environ.get('adminpassw', 'pw')

WTF_CSRF_SECRET_KEY = PASSWORD_STR


app = Flask(__name__)

app.config["DEBUG"] = True
bootstrap = Bootstrap(app)


app.config.update(dict(
    SECRET_KEY=PASSWORD_STR,
    WTF_CSRF_SECRET_KEY=WTF_CSRF_SECRET_KEY
))


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = PdfForm(meta={'csrf': False})

    if form.validate_on_submit():
        f = form.pdff_.data
        filename = secure_filename(f.filename)
        # print(f)
        f.save(os.path.join(
            app.instance_path, filename
        ))
        return redirect(url_for('hello'))

    return render_template('upload.html', form=form)


@app.route('/stealing', methods=['GET'])
def result_table():
