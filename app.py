from flask import Flask, request, jsonify, render_template
from model import predict_digit
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')

        try:
            img = Image.open(file.stream).convert('L').resize((28, 28))
            arr = np.array(img)
        except Exception as e:
            return render_template('upload.html', error=f'Invalid image: {e}')

        pred = predict_digit(arr)
        return render_template('result.html', prediction=pred)

    # GET
    return render_template('upload.html')

@app.route('/predict-file', methods=['POST'])
def predict_file():
    # unchanged API endpoint
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('L').resize((28, 28))
        arr = np.array(img)
    except Exception as e:
        return jsonify({'error': f'Invalid image: {e}'}), 400
    pred = predict_digit(arr)
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
