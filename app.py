from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model
model = load_model('model/model_kopi.h5')

# Label kelas
class_labels = ['Badbeans', 'Longberry', 'Peaberry', 'Premium']

# Fungsi cek ekstensi
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/informasi')
def informasi():
    return render_template('informasi.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocessing
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)[0]
            predicted_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_index]
            confidence = float(prediction[predicted_index])

            return render_template('hasil_deteksi.html',
                                   filename=filename,
                                   label=predicted_class,
                                   confidence=confidence)
    return render_template('upload.html')

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        image_data = data['image'].split(',')[1]
        img_data = base64.b64decode(image_data)
        image_file = BytesIO(img_data)
        img = Image.open(image_file).convert("RGB")

        # Simpan gambar kamera
        debug_filename = 'debug_camera_input.png'
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
        img.save(debug_path)

        # Resize dan prediksi
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        confidence = float(prediction[predicted_index])

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'debug_image_url': url_for('static', filename='uploads/' + debug_filename)
        })

    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)
