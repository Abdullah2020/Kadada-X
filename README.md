# Kadada-X: Rice Leaf Disease Detection

Kadada-X is an AI-powered project designed to detect common diseases in rice leaves using computer vision. It leverages PyTorch to build and train a ResNet18 model for image classification. The goal is to provide a tool that can help in early identification of diseases, potentially aiding in timely agricultural interventions.

Kadada-X was developed as part of 3MTT cohort 3 Knowledge Showcase competition, demonstrating a modular and robust machine learning pipeline.

## Overall System Architecture

The system is designed with distinct modules for data handling, model training, inference, and a web application for user interaction. The code is organized into logical modules for data processing (`data_handling/`), model definition and training (`model_pipeline/`), configuration (`configs/`), and the web interface (`app.py`, `templates/`, `static/`). Pre-trained model weights are stored in `results/saved_models_resnet/`.

You can view the detailed system architecture diagram [here](https://github.com/Abdullah2020/Kadada-X/blob/master/3MTT_system_model1.png) or by clicking the image below:

[![Kadada-X System Architecture](https://raw.githubusercontent.com/Abdullah2020/Kadada-X/master/3MTT_system_model1.png)](https://github.com/Abdullah2020/Kadada-X/blob/master/3MTT_system_model1.png)

## Features

*   **Disease Detection:** Classifies rice leaf images into common disease categories (e.g., Brown Spot, Hispa, Leaf Blast) and Healthy.
*   **PyTorch Backend:** Utilizes PyTorch for building and training a robust CNN model.
*   **Flask Web Application:** Provides an interactive web interface for users to:
    *   Upload rice leaf images.
    *   View the predicted disease and confidence score.
    *   Receive AI-generated management recommendations for the detected disease in either English or Hausa languages.
*   **Multilingual Recommendations:** Leverages Google's Gemini AI to provide agricultural advice in the user's preferred language (English or Hausa).

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python 3.9 or higher
*   `pip` (Python package installer)
*   `git` (for cloning the repository)
*   A virtual environment manager (e.g., `venv`, `conda`) is highly recommended.
*   A Google AI API Key for Gemini (for the recommendation feature). You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation & Setup

1.  **Clone the Repository:**
    Open your terminal or command prompt and run the following command to clone the project files to your local machine:
    ```bash
    git clone https://github.com/Abdullah2020/Kadada-X.git
    ```
    Navigate into the cloned project directory:
    ```bash
    cd Kadada-X
    ```

2.  **Create and Activate a Virtual Environment:**
    It's best practice to use a virtual environment to manage project-specific dependencies. Using `venv` (which comes with Python):
    ```bash
    python -m venv venv
    ```
    Activate the virtual environment:
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    Your terminal prompt should now indicate that you are in the `(venv)` environment.

3.  **Install Dependencies:**
    All required Python libraries are listed in the `requirements.txt` file. Install them by running:
    ```bash
    pip install -r requirements.txt
    ```
  
4.  **Set up Google AI API Key:**
    The recommendation feature relies on the Google Gemini API. You need to provide your API key.
    *   **Recommended Method (Environment Variable):** Set an environment variable named `GOOGLE_API_KEY` with your actual API key.
        *   On Linux/macOS:
            ```bash
            export GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
            ```
        *   On Windows (PowerShell):
            ```bash
            $env:GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
            ```
        *   On Windows (Command Prompt):
            ```bash
            set GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
            ```
        Replace `"YOUR_ACTUAL_API_KEY_HERE"` with your key. The application will automatically pick this up.
    *   *Alternative (Less Secure):* You can directly edit the `app.py` file and replace the placeholder API keys (around lines 40 and 60, where it says `"xxxx-xxxx-xxxxx-xxxxx-"` or similar) with your actual key. This is not recommended for shared or public repositories.

### Running the Application

#### 1. Training the Model (Optional - Pre-trained Model Provided)

This project comes with a pre-trained model (`KadadaX_Vision_v1.pth`) leveraging ResNet18 located in the `results/saved_models_resnet/` directory, so you can skip this step if you just want to run the web application.

However, if you wish to retrain the model (for example, with an updated dataset or different parameters), you can execute the training script:
```bash
python main_training_script.py
```

Before running, ensure your dataset is correctly structured and that the paths and configurations in `configs/base_config.py` are correctly set to point to your dataset. 

#### 2. Running the Web Application (Flask)

To start the Flask web application, which provides the user interface for disease detection and recommendations, run the following command from your project's root directory (while your virtual environment is active):

```bash
python app.py
```

The application will typically start a development server, and you'll see output in your terminal indicating it's running, usually on `http://127.0.0.1:5000/`. Open this URL in your web browser.

You should be greeted by the application's main page, which will look something like this:

[![Kadada-X Web App Interface](https://raw.githubusercontent.com/Abdullah2020/Kadada-X/master/kadadaX_dashboard.png)](https://raw.githubusercontent.com/Abdullah2020/Kadada-X/master/kadadaX_dashboard.png)

**How to Use the Web App:**

1.  On the main page, click the "Choose File" button.
2.  Select a clear image of a rice leaf from your computer that you want to analyze. Sample images for testing can be found in the `other_images/` folder within the repository.
3.  After selecting an image, click the "Upload and Predict" button.
4.  The application will process the image and then display:
    *   The image you uploaded.
    *   The predicted disease name (e.g., "Brown Spot").
    *   The confidence level of the prediction as a percentage.
5.  If a disease is detected, you will see buttons asking if you need management advice. You can choose to receive these recommendations in either **English** or **Hausa**.
6.  Click the button corresponding to your preferred language. The AI-generated agricultural advice will then be displayed on the page.

## Contributing

Contributions to Kadada-X Vision Model are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch for your feature or fix (`git checkout -b feature/YourAmazingFeature` or `git checkout -b bugfix/IssueDescription`).
3.  Make your changes and commit them with clear messages (`git commit -m 'kadadaX need this amazing feature:'`).
4.  Push your changes to your forked repository (`git push origin feature/YourAmazingFeature`).
5.  Open a Pull Request back to the main `Abdullah2020/Kadada-X` repository, clearly describing your changes.


## Acknowledgements

*   This project is a proud contribution to the 3MTT (3 Million Technical Talent) Fellowship program for Knowledge Showcase competition.
