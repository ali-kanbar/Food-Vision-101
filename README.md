# Food Vision 101 🍕🍔🍟

**Food Vision 101** is a deep learning-powered web application that classifies food images into various categories using a pre-trained TensorFlow model. This project combines Flask for the web interface and TensorFlow for the image classification model.

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Food Image Classification**: Upload a food image, and the app will classify it into one of the predefined categories.
- **User-Friendly Interface**: Simple web interface built with Flask for ease of use.
- **TensorFlow Integration**: Pre-trained model loaded and served using TensorFlow(the steps for creating the model are in the .ipynb file).
- **Scalable**: Ready for deployment on cloud platforms like Render.

## Demo
![20240908211156-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/6423832e-b163-410e-8055-810d9f6dd88a)


## Installation

To get this project running locally, follow the steps below.

### Prerequisites
- Python 3.x
- `pip` (Python package installer)
- Git (to clone the repository)

### Clone the Repository
```bash
git clone https://github.com/ali-kanbar/Food-Vision-101.git
cd Food-Vision-101
```

### Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Environment Configuration (Optional)
If you are using environment variables for configurations, ensure to set them in a `.env` file or through your deployment platform.

### Run the Application
```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your web browser to interact with the app.

## Usage

1. Open the web application.
2. Upload an image of food.
3. Wait for the classification result to be displayed, showing the predicted food category.

## Model

The model used in this project is a pre-trained deep learning model from TensorFlow, specifically designed for image classification. It has been fine-tuned to classify different types of food with high accuracy.

### Steps to Update the Model
1. Train or fine-tune a model using a food image dataset.
2. Save the model in the `model/` directory.
3. Load the model in `app.py` for inference during runtime.

## Project Structure
```plaintext
Food-Vision-101/
├── app.py                  # Main Flask application
├── model/                  # Directory for the pre-trained model files
├── static/                 # Static assets (CSS)
├── templates/              # HTML templates for the web interface
├── food_vision_model.ipynb # The jupyter notebook where the model bass created and trained
├── helper_functions.py     # Used functions in the code 
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation (this file)
```

## Technologies Used
- **Flask**: Python-based web framework
- **TensorFlow**: Deep learning framework for the image classification model
- **HTML/CSS**: Front-end design

## Contributing

Contributions are welcome! If you’d like to contribute to this project:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add a new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
