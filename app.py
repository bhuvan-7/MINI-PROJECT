import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = "final_model_resnet50(3).keras"
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}. Ensure the file exists and is accessible. Error: {str(e)}")

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for upload'})

        # Validate file format
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))  # Resize image to 224x224
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make a prediction
            prediction = model.predict(img_array)[0]
            confidence = max(prediction) * 100
            classes = ['Organic', 'Recyclable', 'Unknown']
            predicted_class = classes[np.argmax(prediction)]

            # Classify as "Unknown" if confidence is below the threshold
            confidence_threshold = 95.0  # Adjust this threshold as needed
            if confidence < confidence_threshold:
                predicted_class = "Unknown"

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'file_path': filepath
            })
        else:
            return jsonify({'error': 'Invalid file format. Please upload an image file (png, jpg, jpeg, gif).'})
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})

if __name__ == '__main__':
    # Enable debugging and bind to host 0.0.0.0 for deployment
    port = int(os.environ.get('PORT', 5000))  # Default port is 5000 if not set
    app.run(host="0.0.0.0", port=port, debug=True)
