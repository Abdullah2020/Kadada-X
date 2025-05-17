# ------------------------------------------------------------
# 1. IMPORTS (Flask, your modules, standard libraries)
# ------------------------------------------------------------
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt # Not directly used for web display
import torch
import google.generativeai as genai
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, jsonify # Ensure jsonify is imported

from werkzeug.utils import secure_filename
# from flask import jsonify # Not used in current example, can remove or keep for future APIs

# ------------------------------------------------------------
# 2. PATH SETUP (for your custom modules)
# ------------------------------------------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added '{project_root}' to sys.path")

# --- Import your project-specific modules AFTER path setup ---
try:
    from configs import base_config as cfg
    from model_pipeline.inference_utils import predict_rice_leaf_disease, load_inference_model
    from data_handling.dataset_loader import create_leaf_disease_dataloaders
    print("Project modules imported successfully.")
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# ------------------------------------------------------------
# 3. FLASK APPLICATION INITIALIZATION AND CONFIGURATION
# ------------------------------------------------------------
app = Flask(__name__) # This MUST happen before you define routes with @app.route

# --- Configuration for Flask App ---

ysk = 'xxxx-xxxx-xxxxx-xxxxx-' # <-- Put your real API key here

app.secret_key = ysk  # Needed for flash messages

UPLOAD_FOLDER = os.path.join(project_root, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------------------------------------------
# 4. GLOBAL VARIABLES & ONE-TIME SETUP (like model loading)
# ------------------------------------------------------------
kadada_x_loaded_model = None
gemini_model_instance = None
# cfg.LEAF_DISEASE_CLASSES = [] # This will be populated by load_models_and_config

def load_models_and_config():
    global kadada_x_loaded_model, gemini_model_instance, cfg # Declare we're modifying globals

    # GEMINI API SETUP
    GEMINI_API_KEY = "xxxx-xxxx-xxxxx-xxxxx-"  # <-- Put your real API key here
    if not GEMINI_API_KEY:
        print("Warning: Missing API key. Set GOOGLE_API_KEY environment variable.")
        # Decide if you want to proceed without Gemini or exit
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model_instance = genai.GenerativeModel("gemini-2.0-flash")
            print("Google Generative AI client and model initialized successfully.")
        except Exception as e:
            print(f"Error initializing Google Generative AI client: {e}")
            # gemini_model_instance will remain None

    # --- Load Class Names ---
    if not hasattr(cfg, 'LEAF_DISEASE_CLASSES') or not cfg.LEAF_DISEASE_CLASSES:
        print("cfg.LEAF_DISEASE_CLASSES is not populated. Attempting to load from dataset...")
        try:
            # Make sure DATASET_ROOT_PATH is defined in your base_config.py
            # Or provide a sensible default if it might be missing
            default_dataset_path = os.path.join(project_root, "data", "rice_leaf_diseases_df") # Example path
            cfg.DATASET_ROOT_PATH = getattr(cfg, 'DATASET_ROOT_PATH', default_dataset_path)

            if not os.path.exists(cfg.DATASET_ROOT_PATH):
                raise FileNotFoundError(f"Dataset root path for class names not found: {cfg.DATASET_ROOT_PATH}")

            _, _, discovered_classes = create_leaf_disease_dataloaders(cfg.DATASET_ROOT_PATH)
            cfg.LEAF_DISEASE_CLASSES = discovered_classes
            if not cfg.LEAF_DISEASE_CLASSES:
                raise ValueError("Could not discover classes from dataset.")
            print(f"Successfully loaded class names: {cfg.LEAF_DISEASE_CLASSES}")
        except Exception as e:
            print(f"Warning: Could not automatically load class names: {e}")
            # Provide a fallback list if essential for the app to run
            cfg.LEAF_DISEASE_CLASSES = ["Brown spot", "Hispa", "Leaf blast", "Healthy", "Unknown"]
            print(f"Using fallback class names: {cfg.LEAF_DISEASE_CLASSES}")


    num_classes_for_model = len(cfg.LEAF_DISEASE_CLASSES) if cfg.LEAF_DISEASE_CLASSES else 0
    if num_classes_for_model == 0:
        print("CRITICAL: Number of classes could not be determined. Prediction model might not load correctly.")

    # --- Load the Prediction Model ---
    try:
        # Ensure this model path is correct relative to your project_root
        model_path = os.path.join(project_root, "results", "saved_models_resnet", "KadadaX_Vision_v1.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Prediction model weights file not found at: {model_path}")

        print(f"Attempting to load prediction model for {num_classes_for_model or 'inferred'} classes from {model_path}...")
        kadada_x_loaded_model = load_inference_model(
            model_weights_path=model_path,
            num_classes=num_classes_for_model
        )
        if kadada_x_loaded_model:
            print("Kadada-X prediction model loaded successfully!")
        else:
            print("ERROR: Prediction model could not be loaded (load_inference_model returned None).")
    except Exception as e:
        print(f"ERROR loading prediction model: {e}")
        # kadada_x_loaded_model will remain None

# Call this function once when the app starts, AFTER app is defined
with app.app_context(): # Ensures 'app' is available if needed by extensions during setup
    load_models_and_config()

# ------------------------------------------------------------
# 5. HELPER FUNCTIONS (like allowed_file)
# ------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------------------------------
# 6. FLASK ROUTES (decorated with @app.route)
# ------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def get_uploaded_image(filename):
    # Basic security: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        return "Invalid filename", 400
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    # No need for `global` here if you're just reading `kadada_x_loaded_model` and `cfg`
    # They are already in the global scope of this module.

    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            flash(f'Error saving file: {e}', 'error')
            return redirect(url_for('index'))

        # Check if models and config are loaded
        if not kadada_x_loaded_model:
            flash('Prediction model is not loaded. Please check server logs.', 'error')
            return render_template('result.html', error="Prediction service unavailable.", filename=filename)
        if not cfg.LEAF_DISEASE_CLASSES:
            flash('Disease classes are not loaded. Please check server logs.', 'error')
            return render_template('result.html', error="Prediction configuration missing.", filename=filename)

        try:
            pred_label, pred_conf, _ = predict_rice_leaf_disease(
                image_input=filepath,
                model_instance=kadada_x_loaded_model,
                class_names_list=cfg.LEAF_DISEASE_CLASSES
            )
            pred_label_str = str(pred_label)

            is_healthy_prediction = False
            healthy_keywords = ["healthy", "normal"] # Define your "healthy" class name(s) accurately
            # Ensure LEAF_DISEASE_CLASSES is not empty before list comprehension
            if cfg.LEAF_DISEASE_CLASSES and any(keyword in pred_label_str.lower() for keyword in healthy_keywords if keyword in [cn.lower() for cn in cfg.LEAF_DISEASE_CLASSES]):
                is_healthy_prediction = True
            
            show_rec_buttons = "Error" not in pred_label_str and not is_healthy_prediction

            prediction_data = {
                'label': pred_label_str,
                'confidence': pred_conf,
                'is_healthy': is_healthy_prediction,
                'show_recommendation_buttons': show_rec_buttons
            }
            return render_template('result.html', prediction=prediction_data, filename=filename)

        except Exception as e:
            print(f"Error during prediction: {e}")
            flash(f'Error during prediction: {e}', 'error')
            return render_template('result.html', error=f"Prediction processing error.", filename=filename)
    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg, gif', 'error')
        return redirect(url_for('index'))



import re # Ensure re is imported

def get_english_prompt(disease_name):
    return f"""You are an expert agricultural advisor for rice diseases.
        A rice plant has '{disease_name}'.

        Please provide the advice **in English language**.

        Provide a concise, actionable list of management and prevention strategies for this disease.
        Prioritize methods that are environmentally friendly and sustainable.

        Format the output clearly for web display. Use HTML line breaks (<br>) for paragraphs and HTML lists (<ul><li>...</li></ul>) for bullet points.

        Structure your advice under these categories if possible:
        - Cultural Practices
        - Biological Control
        - Chemical Control (use as last resort)

        The advice should be suitable for small to medium-scale farmers.
        Ensure the output is directly usable as innerHTML. Do NOT include markdown code block delimiters like ```html or ```.
        """

def get_hausa_prompt(disease_name):
    # We are aiming for the AI to produce something like:
    # Fanni:
    # - Batu na daya
    # - Batu na biyu
    #
    # Ko kuma:
    # Fanni:
    # Batu na daya.
    # Batu na biyu.
    # (where each "Batu" is on a new line, which we'll convert to <br>)

    return f"""KA BA DA DUKKAN AMSA CIKIN HARSHEN HAUSA KAWAI. KADA KA YI AMFANI DA TURANCI KO WANI YARE.
        Kai ƙwararren mai ba da shawara ne kan harkokin noma, musamman cututtukan shinkafa.
        An gano shukar shinkafa tana da cutar '{disease_name}'.

        Domin nuna shawarwarin a shafin intanet yadda ya kamata:
        1.  Kowane babban fanni (kamar Ayyukan Noma) ya kasance da taken sa a layi daban, sannan ka sa alamar digo biyu (:). Misali: "Ayyukan Noma na Gargajiya:".
        2.  A karkashin kowane fanni, kowane karamin batu ko shawara ya kasance a layin sa daban. Fara kowane karamin batu da alamar ƙwayar tauraro da sarari (misali, "* Shawara ta farko...").
        3.  Kada ka yi amfani da wasu alamomin HTML kamar <ul>, <li>, ko <p>. Kawai ka rubuta rubutu kai tsaye tare da tsarin da aka ambata a sama.

        Ka ba da jerin gajeru kuma masu amfani na dabarun magance wannan cuta da matakan kariya.
        Ka fi bada fifiko ga hanyoyin da ba sa cutar da muhalli kuma masu dorewa.

        Ka tsara shawarwarin a ƙarƙashin waɗannan fannoni, kana bin tsarin da aka ambata a sama:
        - Ayyukan Noma na Gargajiya
        - Magungunan Halittu Masu Rai
        - Magungunan Sinadarai (a yi amfani da su a karshe idan babu yadda za a yi)

        Shawarwarin su dace da ƙanana da matsakaitan manoma.
        KA TABBATA DUK RUBUTUN AMSA YANA CIKIN HARSHEN HAUSA KAWAI.
        KADA KA SAKA ALAMOMIN MARKDOWN NA CODE BLOCK KAMAR ```html KO ```.
        """



@app.route('/api/get_recommendation', methods=['POST'])
def api_get_recommendation():
    global gemini_model_instance # Assuming this is loaded globally
    
    data = request.get_json()
    if not data or 'disease_name' not in data or 'language' not in data:
        return jsonify({'status': 'error', 'error': 'Disease name or language not provided.'}), 400

    disease_name = data['disease_name']
    selected_language = data['language']

    if not gemini_model_instance:
        return jsonify({'status': 'error', 'error': 'AI Recommendation service not available.'}), 503
    
    prompt_content = ""
    if selected_language.lower() == 'hausa':
        prompt_content = get_hausa_prompt(disease_name)
    else: # Default to English
        prompt_content = get_english_prompt(disease_name)

    try:
        print(f"API: Requesting AI recommendation for: {disease_name} in {selected_language}")
        print(f"----\nPROMPT:\n{prompt_content}\n----") # Print the exact prompt for debugging
        response = gemini_model_instance.generate_content(prompt_content)
        
        cleaned_text = response.text.strip()
        # Remove potential markdown delimiters (important fallback)
        cleaned_text = re.sub(r'^```html\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
        cleaned_text = cleaned_text.strip()

        # If Hausa output is plain text with newlines, and AI didn't use <br>
        # Convert newlines to <br> for HTML display
        # This is a simple approach; more robust might involve checking if HTML tags are already present.
        if selected_language.lower() == 'hausa' and '<ul' not in cleaned_text.lower() and '<p>' not in cleaned_text.lower():
            cleaned_text = cleaned_text.replace('\n', '<br>')

        return jsonify({
            'status': 'success',
            'disease_name': disease_name,
            'language_used': selected_language,
            'recommendation_text': cleaned_text
        })
    except Exception as e:
        print(f"API Error fetching AI recommendation: {e}")
        import traceback
        print(traceback.format_exc()) # Good for detailed debugging
        return jsonify({'status': 'error', 'error': f'AI Recommendation processing error: {str(e)}'}), 500



# ------------------------------------------------------------
# 7. MAIN EXECUTION BLOCK (to run the development server)
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True) # debug=True is for development. Set to False for production.