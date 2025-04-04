import streamlit as st
import os
from PIL import Image
import io
import time  # To measure processing time
from datetime import datetime, timedelta  # For forecast time handling
import cv2  # For OpenCV blur detection
import numpy as np  # For OpenCV image processing
import google.generativeai as genai  # Import Google Gemini library
import requests  # For OpenWeather API calls
import json  # For parsing JSON

# --- Constants ---
PLANT_VILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

DISEASE_RISK_RULES = {
    "Tomato": {
        "Late Blight": {
            "conditions": [
                {"param": "temp", "min": 10, "max": 25, "hours_min": 6},
                {"param": "humidity", "min": 75, "hours_min": 6}
            ],
            "message": "High humidity and moderate temperatures increase the risk of Late Blight."
        },
        "Early Blight": {
            "conditions": [
                {"param": "temp", "min": 20, "max": 30, "hours_min": 4},
                {"param": "humidity", "min": 80, "hours_min": 4}
            ],
            "message": "Warm, humid conditions favor Early Blight development."
        }
    },
    "Rose": {
        "Powdery Mildew": {
            "conditions": [
                {"param": "temp", "min": 15, "max": 28, "hours_min": 6},
                {"param": "humidity", "min": 70, "max": 90, "hours_min": 6}
            ],
            "message": "Moderate temperatures and high (but not raining) humidity favor Powdery Mildew."
        },
        "Black Spot": {
            "conditions": [
                {"param": "humidity", "min": 85, "hours_min": 7},
                {"param": "temp", "min": 18, "max": 26, "hours_min": 7}
            ],
            "message": "Prolonged leaf wetness and warm temperatures increase Black Spot risk."
        }
    },
    "Apple": {
        "Apple Scab": {
            "conditions": [
                {"param": "temp", "min": 6, "max": 24, "hours_min": 9},
                {"param": "rain", "min": 0.1, "hours_min": 9}
            ],
            "message": "Cool, wet conditions during spring favor Apple Scab infection."
        }
    }
}

PLANT_OPTIONS = ["Select Plant...", "Tomato", "Rose", "Apple"]

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Plant Health Assistant",
    page_icon="🧑‍🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Title ---
st.title("🧑‍🌾 AI Plant Health Assistant")
st.caption("Upload a plant image. AI will identify diseases, pests, or weeds and provide details & weather context.")

# --- API Key Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    if not GOOGLE_API_KEY:
        st.error("Google API Key (GOOGLE_API_KEY) not found or empty in .streamlit/secrets.toml.")
        st.stop()
    if not OPENWEATHER_API_KEY:
        st.error("OpenWeather API Key (OPENWEATHER_API_KEY) not found or empty in .streamlit/secrets.toml.")
        st.stop()
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError as e:
    st.error(f"API Key '{e}' not found in .streamlit/secrets.toml. Please ensure it's correctly named.")
    st.stop()
except FileNotFoundError:
    st.error(".streamlit/secrets.toml file not found. Please create it and add your API keys.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during API key configuration: {e}")
    st.stop()

# --- Model Initialization ---
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

try:
    vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    text_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Failed to initialize Gemini models: {e}")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Analysis & Alert Options")
user_city = st.sidebar.text_input("Enter City Name (for Weather):", help="e.g., Mumbai, London, New York")
selected_plant = st.sidebar.selectbox("Select Plant for Disease Alerts:", options=PLANT_OPTIONS, help="Choose the plant type you want weather-based risk alerts for.")

# --- Helper Functions ---
def check_blurriness(image_bytes, filename, threshold=100.0):
    """Calculates Laplacian variance to detect blurriness."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return 0.0, False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance, variance < threshold
    except Exception:
        return 0.0, False

def get_weather_data(city, api_key):
    """Fetches current weather data from OpenWeatherMap."""
    if not city:
        return {"data": None, "error": "No city provided."}
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city + "&units=metric"
    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
            return {"data": None, "error": data.get("message", "City not found or API error.")}
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        return {
            "data": {
                "city": data.get("name"),
                "temp": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "description": weather.get("description"),
                "icon": weather.get("icon"),
                "wind_speed": wind.get("speed"),
            },
            "error": None
        }
    except requests.exceptions.RequestException as e:
        return {"data": None, "error": f"Weather API request failed: {e}"}
    except Exception as e:
        return {"data": None, "error": f"Error processing weather data: {e}"}

def get_weather_forecast(city, api_key):
    """Fetches 5-day/3-hour weather forecast from OpenWeatherMap."""
    if not city:
        return {"data": None, "error": "No city provided."}
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city + "&units=metric"
    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()
        if str(data.get("cod")) != "200":
            return {"data": None, "error": data.get("message", "City not found or API error.")}
        return {"data": data.get("list", []), "error": None}
    except requests.exceptions.RequestException as e:
        return {"data": None, "error": f"Forecast API request failed: {e}"}
    except Exception as e:
        return {"data": None, "error": f"Error processing forecast data: {e}"}

def check_disease_risk(plant_type, forecast_data, rules, forecast_window_hours=48):
    """Checks forecast data against disease rules for a specific plant."""
    alerts = []
    if not plant_type or plant_type == "Select Plant..." or not forecast_data:
        return alerts
    plant_rules = rules.get(plant_type, {})
    if not plant_rules:
        return alerts
    now = datetime.now()
    forecast_end_time = now + timedelta(hours=forecast_window_hours)
    relevant_forecasts = [
        point for point in forecast_data
        if now <= datetime.fromtimestamp(point.get("dt", 0)) <= forecast_end_time
    ]
    for disease, rule in plant_rules.items():
        conditions_met_count = 0
        for condition in rule.get("conditions", []):
            param = condition.get("param")
            min_val = condition.get("min")
            max_val = condition.get("max")
            hours_min = condition.get("hours_min", 1)
            hours_met = 0
            for i, point in enumerate(relevant_forecasts):
                value = None
                if param == "temp": value = point.get("main", {}).get("temp")
                elif param == "humidity": value = point.get("main", {}).get("humidity")
                elif param == "rain": value = point.get("rain", {}).get("3h", 0)
                if value is not None:
                    is_met = True
                    if min_val is not None and value < min_val: is_met = False
                    if max_val is not None and value > max_val: is_met = False
                    if is_met:
                        hours_met += 3
                else:
                    hours_met = 0
                if hours_met >= hours_min:
                    conditions_met_count += 1
        if conditions_met_count == len(rule.get("conditions", [])):
            alerts.append(f"**{disease} ({plant_type}):** {rule.get('message', 'Conditions met.')}")
    return alerts

def get_plant_identification(image_bytes, filename, is_reanalysis=False, initial_guess=None):
    """Calls Gemini Vision to identify plant disease, pest, or weed."""
    analysis_start_time = time.time()
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt = (
            f"Re-analyze the provided plant image. The initial analysis suggested '{initial_guess}', "
            f"but user feedback indicates this might be inaccurate. Provide the single most likely identification."
            if is_reanalysis else
            f"Analyze the provided image of a plant. Identify the most likely issue: disease, pest, weed, or healthy."
        )
        response = vision_model.generate_content([prompt, img], stream=False, safety_settings=safety_settings)
        response.resolve()
        identified_issue = response.text.strip()
        analysis_end_time = time.time()
        return {
            "identified_issue": identified_issue,
            "error": None,
            "processing_time": analysis_end_time - analysis_start_time
        }
    except Exception as e:
        error_msg = f"Gemini Vision API call failed for {filename}: {e}"
        st.error(error_msg)
        return {"identified_issue": "Error", "error": error_msg, "processing_time": time.time() - analysis_start_time}

def get_issue_details(issue_name, filename):
    """Calls Gemini Text model to get details about the identified issue."""
    if not issue_name or issue_name in ["Unknown/Not Plant", "Error", "Healthy Plant"]:
        return {"details": "No specific details needed for this classification.", "error": None}
    analysis_start_time = time.time()
    try:
        prompt = f"Provide detailed information about the plant issue: '{issue_name}'. Include sections like Type, Symptoms, Cause, Management, and Prevention."
        response = text_model.generate_content(prompt, stream=False, safety_settings=safety_settings)
        response.resolve()
        details = response.text
        analysis_end_time = time.time()
        return {
            "details": details,
            "error": None,
            "processing_time": analysis_end_time - analysis_start_time
        }
    except Exception as e:
        error_msg = f"Gemini Text API call failed for details on {issue_name}: {e}"
        st.error(error_msg)
        return {"details": "Error retrieving details.", "error": error_msg, "processing_time": time.time() - analysis_start_time}

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Choose plant images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or more images of plants for analysis."
)

if uploaded_files:
    st.write(f"You uploaded {len(uploaded_files)} image(s).")
else:
    st.info("Select a plant and enter city for alerts. Upload images for analysis.")

# --- Weather Forecast & Disease Alert Section ---
st.markdown("---")
st.subheader("🌦️ Weather & Disease Risk Alerts")
if user_city and selected_plant != "Select Plant...":
    with st.spinner(f"Fetching forecast for {user_city} and checking risks for {selected_plant}..."):
        forecast_info = get_weather_forecast(user_city, OPENWEATHER_API_KEY)
        if forecast_info["error"]:
            st.warning(f"Could not fetch forecast data for {user_city}: {forecast_info['error']}")
            disease_alerts = []
        else:
            disease_alerts = check_disease_risk(selected_plant, forecast_info["data"], DISEASE_RISK_RULES)
        if disease_alerts:
            st.warning("Potential Disease Risks Based on Forecast:")
            for alert in disease_alerts:
                st.markdown(f"- {alert}")
        else:
            st.success(f"Forecast conditions do not indicate high risk for common diseases for {selected_plant} in {user_city} in the next 48 hours.")
elif user_city and selected_plant == "Select Plant...":
    st.info("Select a plant type from the sidebar to get disease risk alerts.")
elif not user_city:
    st.info("Enter a city name in the sidebar to get weather forecasts and disease risk alerts.")

# --- Analysis Trigger ---
st.markdown("---")
analyze_button = st.button("Analyze Uploaded Images", type="primary", disabled=not uploaded_files)

# --- Session State Initialization ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# --- Analysis Execution ---
if analyze_button and uploaded_files:
    st.session_state.analysis_results = {}
    start_time_total = time.time()
    progress_bar = st.progress(0, text="Starting analysis...")
    weather_info = {"data": None, "error": None}
    if user_city:
        weather_info = get_weather_data(user_city, OPENWEATHER_API_KEY)
    with st.spinner("Analyzing images... This may take a moment."):
        num_files = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            image_bytes = uploaded_file.getvalue()
            current_result = {"weather": weather_info}
            base_progress = (i / num_files)
            max_progress = ((i + 1) / num_files)
            # --- Blur Check ---
            progress_bar.progress(base_progress + 0.1 * (1/num_files), text=f"({i+1}/{num_files}) Checking blur: {filename}")
            blur_variance, is_blurry = check_blurriness(image_bytes, filename)
            current_result["blur_info"] = {"variance": blur_variance, "is_blurry": is_blurry}
            # --- Step 1: Identification ---
            progress_bar.progress(base_progress + 0.3 * (1/num_files), text=f"({i+1}/{num_files}) Identifying: {filename}")
            identification_result = get_plant_identification(image_bytes, filename)
            current_result["identification"] = identification_result
            identified_issue = identification_result.get("identified_issue", "Error")
            # --- Step 2: Get Details ---
            details_result = {"details": "Skipped.", "error": None, "processing_time": 0}
            if identified_issue not in ["Unknown/Not Plant", "Error", "Healthy Plant"]:
                progress_bar.progress(base_progress + 0.7 * (1/num_files), text=f"({i+1}/{num_files}) Getting details for {identified_issue[:20]}...")
                details_result = get_issue_details(identified_issue, filename)
            else:
                details_result["details"] = "No specific details needed for this classification."
            current_result["details"] = details_result
            st.session_state.analysis_results[filename] = current_result
            progress_bar.progress(max_progress, text=f"Finished: {filename}")
            time.sleep(0.05)
    progress_bar.empty()
    total_time = time.time() - start_time_total
    st.success(f"Image analysis complete for {num_files} images in {total_time:.2f} seconds.")

# --- Display Analysis Results ---
st.markdown("---")
st.subheader("📊 Image Analysis Results")
if not st.session_state.analysis_results and not analyze_button:
    st.info("Upload images and click 'Analyze Uploaded Images' to see results.")
elif not st.session_state.analysis_results and analyze_button:
    st.warning("Analysis was run, but no results were generated. Check for errors above.")
else:
    image_filenames = list(st.session_state.analysis_results.keys())
    tab_titles = [f"{fname[:20]}{'...' if len(fname)>20 else ''}" for fname in image_filenames]
    tabs = st.tabs(tab_titles)
    for i, tab in enumerate(tabs):
        with tab:
            filename = image_filenames[i]
            result = st.session_state.analysis_results[filename]
            col1, col2 = st.columns([1, 2])
            with col1:
                uploaded_file_obj = next((f for f in uploaded_files if f.name == filename), None)
                if uploaded_file_obj:
                    try:
                        img_display = Image.open(uploaded_file_obj)
                        st.image(img_display, caption=filename, use_container_width=True)
                    except Exception as img_err:
                        st.warning(f"Could not display image {filename}: {img_err}")
                else:
                    st.caption(f"Image data for {filename} not available for display.")
                blur_info = result.get("blur_info")
                if blur_info and blur_info["is_blurry"]:
                    st.warning(f"⚠️ Possible Blur (Variance: {blur_info['variance']:.2f})")
                weather = result.get("weather", {})
                weather_data = weather.get("data")
                if weather_data:
                    st.markdown("**Current Weather Context:**")
                    icon_url = f"http://openweathermap.org/img/wn/{weather_data.get('icon')}@2x.png" if weather_data.get('icon') else None
                    if icon_url: st.image(icon_url, width=50)
                    st.markdown(f"**{weather_data.get('city', 'N/A')}:** {weather_data.get('description', 'N/A').capitalize()}")
                    st.markdown(f"**Temp:** {weather_data.get('temp', 'N/A')}°C (Feels like: {weather_data.get('feels_like', 'N/A')}°C)")
                    st.markdown(f"**Humidity:** {weather_data.get('humidity', 'N/A')}%")
                    st.markdown(f"**Wind:** {weather_data.get('wind_speed', 'N/A')} m/s")
                elif user_city and weather.get("error"):
                    st.caption(f"Current Weather Error: {weather.get('error')}")
            with col2:
                identification = result.get("identification", {})
                identified_issue = identification.get("identified_issue", "Error")
                id_error = identification.get("error")
                id_time = identification.get("processing_time", 0)
                st.markdown("**Identification:**")
                if id_error:
                    st.error(f"Error: {id_error}")
                elif identified_issue == "Unknown/Not Plant":
                    st.info("❓ Unknown / Not a Plant")
                elif identified_issue == "Healthy Plant":
                    st.success("✅ Healthy Plant")
                elif identified_issue == "Error":
                    st.error("❗️ Identification Error")
                else:
                    st.warning(f"⚠️ Identified Issue: **{identified_issue}**")
                st.caption(f"ID Time: {id_time:.2f}s")
                feedback_options = ["Not Rated", "Accurate", "Inaccurate", "Partially Accurate", "Unsure"]
                feedback_key = f"feedback_{filename}"
                previous_feedback = st.session_state.feedback.get(filename, "Not Rated")
                try: current_index = feedback_options.index(previous_feedback)
                except ValueError: current_index = 0
                user_feedback = st.radio(
                    "Rate Analysis Quality:", options=feedback_options, index=current_index,
                    key=feedback_key, horizontal=True
                )
                reanalysis_result_display = None
                trigger_states = ["Inaccurate", "Partially Accurate", "Unsure"]
                if user_feedback in trigger_states and user_feedback != previous_feedback:
                    st.session_state.feedback[filename] = user_feedback
                    with st.spinner(f"Re-analyzing {filename} based on feedback..."):
                        reanalysis_image_bytes = None
                        if uploaded_files:
                            reanalysis_file_obj = next((f for f in uploaded_files if f.name == filename), None)
                            if reanalysis_file_obj:
                                reanalysis_image_bytes = reanalysis_file_obj.getvalue()
                        if reanalysis_image_bytes and identified_issue not in ["Error", "Unknown/Not Plant", "Healthy Plant"]:
                            reanalysis_result = get_plant_identification(
                                reanalysis_image_bytes,
                                filename,
                                is_reanalysis=True,
                                initial_guess=identified_issue
                            )
                            st.session_state.analysis_results[filename]['reanalysis'] = reanalysis_result
                            reanalysis_result_display = reanalysis_result
                        else:
                            st.warning("Could not perform re-analysis (image data missing or initial ID not suitable).")
                    st.rerun()
                elif user_feedback != previous_feedback:
                    st.session_state.feedback[filename] = user_feedback
                    if 'reanalysis' in st.session_state.analysis_results[filename]:
                        del st.session_state.analysis_results[filename]['reanalysis']
                stored_reanalysis = result.get('reanalysis')
                if stored_reanalysis and user_feedback in trigger_states:
                    reanalysis_result_display = stored_reanalysis
                if reanalysis_result_display:
                    re_identified_issue = reanalysis_result_display.get("identified_issue", "Error")
                    re_id_error = reanalysis_result_display.get("error")
                    re_id_time = reanalysis_result_display.get("processing_time", 0)
                    st.markdown("**Alternative Suggestion (Re-analysis):**")
                    if re_id_error:
                        st.error(f"Re-analysis Error: {re_id_error}")
                    elif re_identified_issue == identified_issue:
                        st.info(f"Re-analysis confirmed: **{re_identified_issue}**")
                    elif re_identified_issue == "Unknown/Not Plant":
                        st.info("❓ Re-analysis: Unknown / Not a Plant")
                    elif re_identified_issue == "Healthy Plant":
                        st.success("✅ Re-analysis: Healthy Plant")
                    elif re_identified_issue == "Error":
                        st.error("❗️ Re-analysis Error")
                    else:
                        st.warning(f"⚠️ Alternative: **{re_identified_issue}**")
                    st.caption(f"Re-ID Time: {re_id_time:.2f}s")
                details_info = result.get("details", {})
                details_text = details_info.get("details", "N/A")
                details_error = details_info.get("error")
                details_time = details_info.get("processing_time", 0)
                st.markdown(f"**Details & Management (for {identified_issue}):**")
                if details_error:
                    st.error(f"Details Error: {details_error}")
                else:
                    st.markdown(details_text)
                if details_time > 0:
                    st.caption(f"Details Time: {details_time:.2f}s")

# --- Sidebar Bottom ---
st.sidebar.divider()
st.sidebar.header("ℹ️ Info & Feedback")
feedback_summary = [{"Image": fname, "Feedback": fback} for fname, fback in st.session_state.feedback.items() if fback != "Not Rated"]
if feedback_summary:
    st.sidebar.dataframe(feedback_summary, use_container_width=True)
else:
    st.sidebar.caption("No feedback provided yet.")
st.sidebar.divider()
with st.sidebar.expander("How to Use"):
    st.markdown("""
    1. **API Keys:** Ensure `GOOGLE_API_KEY` and `OPENWEATHER_API_KEY` are in `.streamlit/secrets.toml`.
    2. **Enter City:** Provide your city name for weather context and alerts.
    3. **Select Plant:** Choose a plant type for disease risk alerts.
    4. **Upload Images:** Upload one or more plant images for analysis.
    5. **Analyze:** Click 'Analyze Uploaded Images'.
    6. **View Results:** Check the main area for alerts and the tabs for image-specific analysis.
    7. **Provide Feedback:** Rate the analysis quality using the radio buttons under each image result.
    """)
st.sidebar.divider()
st.sidebar.caption("Powered by Google Gemini & OpenWeatherMap.")
