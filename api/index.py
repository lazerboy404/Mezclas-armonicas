from flask import Flask, render_template, request, jsonify
import os
import audio_manager
import json
import hashlib
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# Inicializar Firebase
cred = credentials.Certificate("firebase-service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Tu código actual de web_app.py aquí...
# [Pega todo el contenido de web_app.py desde la línea 18 en adelante]

# Mover las rutas de templates al final
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)