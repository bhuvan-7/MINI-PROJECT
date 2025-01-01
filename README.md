# Waste Classification Model: Organic, Recyclable, and Unknown

This repository contains the code and resources for a deep learning-based waste classification system. The model classifies waste images into three categories: **Organic**, **Recyclable**, and **Unknown**, leveraging the ResNet50 architecture. A user-friendly web interface is provided for real-time waste classification.

---

## Features

1. **Deep Learning Model**:
   - Trained using the ResNet50 architecture.
   - High accuracy achieved in classifying diverse waste types.

2. **Web Application**:
   - Flask-based web interface for easy interaction.
   - Allows users to upload waste images and receive predictions.

3. **Robust Classification**:
   - Low-confidence predictions (<99.6%) are categorized as "Unknown."

4. **Scalable Deployment**:
   - Ready to deploy on platforms like Render.

---

## Folder Structure

```
├── app.py                # Main application script
├── requirements.txt      # Python dependencies
├── Procfile              # For deployment on Render
├── static/
│   └── uploads/          # Directory for uploaded images
├── templates/
│   └── index.html        # HTML template for the web interface
├── final_model_resnet50(3).keras  # Trained model file
└── README.md             # Project documentation
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the Model File is Available**:
   - Make sure `final_model_resnet50(3).keras` is present in the root directory.
   - If hosted externally, ensure the download logic in `app.py` is correct.

4. **Run the Application Locally**:
   ```bash
   python app.py
   ```
   The app will be accessible at `http://127.0.0.1:5000/`.

5. **Deploy on Render**:
   - Push the repository to GitHub.
   - Connect the repository to Render and follow the deployment steps.

---

## Usage

1. Access the web app.
2. Upload an image of waste.
3. Receive the classification result along with confidence levels.

---

## Deployment on Render

1. Ensure `requirements.txt` and `Procfile` are correctly set up.
2. If the model file is large, host it externally and use code to download it during deployment.
3. Add the following environment variables in Render:
   - `UPLOAD_FOLDER` set to `static/uploads`

---

## Future Enhancements

- **Additional Waste Categories**: Extend classification to include electronic and hazardous waste.
- **IoT Integration**: Connect with smart waste bins for automated waste sorting.
- **Mobile Accessibility**: Deploy the model on mobile platforms.
- **Multilingual Support**: Enhance usability for a global audience.

---

## Troubleshooting

### Common Issues
1. **Model Not Found**:
   - Ensure the model file is in the root directory or hosted externally with proper download logic.

2. **Port Binding Error on Render**:
   - Add the following line in `app.py`:
     ```python
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
     ```

3. **Git LFS Quota Exceeded**:
   - Untrack the model file from Git LFS and commit it normally.

---

## Acknowledgments

- **Keras**: For providing the deep learning framework.
- **Flask**: For simplifying web application development.
- **Render**: For deployment services.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

