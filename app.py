import streamlit as st
import os
from PIL import Image
import io
import time # To measure processing time
from datetime import datetime, timedelta # Added for forecast time handling
import cv2 # For OpenCV blur detection
import numpy as np # For OpenCV image processing
import google.generativeai as genai # Import Google Gemini library
import requests # For OpenWeather API calls
import json # For parsing JSON

# --- Constants ---
# A sample list of PlantVillage classes to guide the model.
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

# --- Disease Risk Rules (Example Structure) ---
# Conditions based on OpenWeatherMap forecast data (temp in C, humidity in %, rain in mm)
# We check conditions within a certain forecast window (e.g., next 48 hours)
# This is a basic example, more complex rules can be added.
DISEASE_RISK_RULES = {
    "Tomato": {
        "Late Blight": {
            "conditions": [
                {"param": "temp", "min": 10, "max": 25, "hours_min": 6}, # Temp between 10-25C for >= 6 hours
                {"param": "humidity", "min": 75, "hours_min": 6}        # Humidity >= 75% for >= 6 hours
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
                 {"param": "humidity", "min": 70, "max": 90, "hours_min": 6} # High, but not saturated
            ],
            "message": "Moderate temperatures and high (but not raining) humidity favor Powdery Mildew."
        },
         "Black Spot": {
             "conditions": [
                 {"param": "humidity", "min": 85, "hours_min": 7}, # Needs leaf wetness
                 {"param": "temp", "min": 18, "max": 26, "hours_min": 7}
             ],
             "message": "Prolonged leaf wetness and warm temperatures increase Black Spot risk."
         }
    },
    "Apple": {
        "Apple Scab": {
            "conditions": [
                {"param": "temp", "min": 6, "max": 24, "hours_min": 9}, # Wide temp range
                {"param": "rain", "min": 0.1, "hours_min": 9} # Needs rain/wetness duration
            ],
            "message": "Cool, wet conditions during spring favor Apple Scab infection."
        }
    }
    # Add rules for other plants and diseases
}

# List of plants for the dropdown
PLANT_OPTIONS = ["Select Plant...", "Tomato", "Rose", "Apple"] # Add more as rules are defined

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
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"] # Read OpenWeather key
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
# Location Input
user_city = st.sidebar.text_input("Enter City Name (for Weather):", help="e.g., Mumbai, London, New York")
# Plant Selection for Alerts
selected_plant = st.sidebar.selectbox("Select Plant for Disease Alerts:", options=PLANT_OPTIONS, help="Choose the plant type you want weather-based risk alerts for.")

# --- Helper Functions ---

def check_blurriness(image_bytes, filename, threshold=100.0):
    """Calculates Laplacian variance to detect blurriness."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: return 0.0, False # Cannot check blur if decode fails
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance, variance < threshold
    except Exception:
        return 0.0, False # Default to not blurry on error

def get_weather_data(city, api_key):
    """Fetches current weather data from OpenWeatherMap."""
    if not city:
        return {"data": None, "error": "No city provided."}
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city + "&units=metric" # Use metric units
    try:
        response = requests.get(complete_url)
        # st.write(f"DEBUG: Current Weather API Status Code: {response.status_code}") # Optional Debug
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
             error_message = data.get("message", "City not found or API error.")
             # st.write(f"DEBUG: Current Weather API Error Response: {data}") # Optional Debug
             return {"data": None, "error": error_message}
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
        # st.write(f"DEBUG: Current Weather Request Exception: {e}") # Optional Debug
        return {"data": None, "error": f"Weather API request failed: {e}"}
    except Exception as e:
        # st.write(f"DEBUG: Current Weather Processing Exception: {e}") # Optional Debug
        return {"data": None, "error": f"Error processing weather data: {e}"}

def get_weather_forecast(city, api_key):
    """Fetches 5-day/3-hour weather forecast from OpenWeatherMap."""
    if not city:
        return {"data": None, "error": "No city provided."}
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city + "&units=metric"
    try:
        response = requests.get(complete_url)
        st.write(f"DEBUG: Forecast API Status Code: {response.status_code}") # DEBUG: Show status code
        response.raise_for_status()
        data = response.json()
        if str(data.get("cod")) != "200": # API returns cod as string '200' for forecast
             error_message = data.get("message", "City not found or API error.")
             st.write(f"DEBUG: Forecast API Error Response: {data}") # DEBUG: Show error response
             return {"data": None, "error": error_message}
        forecast_list = data.get("list", [])
        st.write(f"DEBUG: Received {len(forecast_list)} forecast points.") # DEBUG: Show count
        st.write("DEBUG: Raw Forecast Data Snippet (first 2):", forecast_list[:2]) # DEBUG: Show first 2 forecast points
        return {"data": forecast_list, "error": None} # Return the list of forecast points
    except requests.exceptions.RequestException as e:
        st.error(f"DEBUG: Forecast Request Exception: {e}") # DEBUG: Show error
        return {"data": None, "error": f"Forecast API request failed: {e}"}
    except Exception as e:
        st.error(f"DEBUG: Forecast Processing Exception: {e}") # DEBUG: Show error
        return {"data": None, "error": f"Error processing forecast data: {e}"}

def check_disease_risk(plant_type, forecast_data, rules, forecast_window_hours=48):
    """Checks forecast data against disease rules for a specific plant."""
    alerts = []
    st.write(f"DEBUG: Checking disease risk for plant: {plant_type}") # DEBUG
    if not plant_type or plant_type == "Select Plant..." or not forecast_data:
        st.write("DEBUG: Skipping risk check (no plant selected or no forecast data).") # DEBUG
        return alerts

    plant_rules = rules.get(plant_type, {})
    if not plant_rules:
        st.write(f"DEBUG: No rules defined for plant: {plant_type}") # DEBUG
        return alerts # No rules defined for this plant

    now = datetime.now()
    forecast_end_time = now + timedelta(hours=forecast_window_hours)

    # Filter forecast points within the window
    relevant_forecasts = [
        point for point in forecast_data
        if now <= datetime.fromtimestamp(point.get("dt", 0)) <= forecast_end_time
    ]
    st.write(f"DEBUG: Found {len(relevant_forecasts)} relevant forecast points in the next {forecast_window_hours} hours.") # DEBUG

    if not relevant_forecasts:
        return alerts

    for disease, rule in plant_rules.items():
        st.write(f"DEBUG: Checking rule for disease: {disease}") # DEBUG
        conditions_met_count = 0
        all_conditions_details = [] # Store details for debugging

        for condition_index, condition in enumerate(rule.get("conditions", [])):
            param = condition.get("param")
            min_val = condition.get("min")
            max_val = condition.get("max")
            hours_min = condition.get("hours_min", 1) # Minimum consecutive hours condition must be met
            condition_str = f"Cond {condition_index+1} ({param}{f'>={min_val}' if min_val is not None else ''}{f'<={max_val}' if max_val is not None else ''} for {hours_min}h)" # Debug string

            hours_met = 0
            max_consecutive_hours = 0

            for i, point in enumerate(relevant_forecasts):
                value = None
                dt_str = datetime.fromtimestamp(point.get("dt", 0)).strftime('%Y-%m-%d %H:%M')
                point_debug_str = f"Point {i} ({dt_str}): "

                if param == "temp": value = point.get("main", {}).get("temp")
                elif param == "humidity": value = point.get("main", {}).get("humidity")
                elif param == "rain": value = point.get("rain", {}).get("3h", 0) # Rain in last 3h

                if value is not None:
                    is_met = True
                    if min_val is not None and value < min_val: is_met = False
                    if max_val is not None and value > max_val: is_met = False
                    point_debug_str += f"{param}={value}, Met={is_met}. "

                    if is_met:
                        time_diff_hours = 3 if i > 0 else 3 # Assume 3h interval
                        hours_met += time_diff_hours
                    else:
                        max_consecutive_hours = max(max_consecutive_hours, hours_met)
                        hours_met = 0 # Reset counter if condition breaks
                else:
                     point_debug_str += f"{param}=N/A, Met=False. "
                     max_consecutive_hours = max(max_consecutive_hours, hours_met)
                     hours_met = 0 # Reset if data missing
                # st.write(f"   {point_debug_str} CurrentConsecutive={hours_met}h") # DEBUG: Very verbose point check

            max_consecutive_hours = max(max_consecutive_hours, hours_met) # Check final sequence
            condition_met_final = max_consecutive_hours >= hours_min
            all_conditions_details.append(f"{condition_str}: {'MET' if condition_met_final else 'NOT MET'} (Max consecutive: {max_consecutive_hours}h)") # Debug detail

            if condition_met_final:
                conditions_met_count += 1

        st.write(f"DEBUG: Disease '{disease}' Condition Results: {'; '.join(all_conditions_details)}") # DEBUG: Show summary for disease

        # If all conditions for the disease are met
        if conditions_met_count == len(rule.get("conditions", [])):
            alerts.append(f"**{disease} ({plant_type}):** {rule.get('message', 'Conditions met.')}")

    st.write(f"DEBUG: Final generated alerts: {alerts}") # DEBUG
    st.write(f"DEBUG: Final generated alerts: {alerts}") # DEBUG
    return alerts


def get_plant_identification(image_bytes, filename, is_reanalysis=False, initial_guess=None):
    """Calls Gemini Vision to identify plant disease, pest, or weed.
       Can optionally perform re-analysis based on initial guess."""
    analysis_start_time = time.time()
    try:
        img = Image.open(io.BytesIO(image_bytes))

        if is_reanalysis:
            prompt = f"""Re-analyze the provided plant image.
            The initial analysis suggested '{initial_guess}', but user feedback indicates this might be inaccurate or uncertain.
            Please carefully re-examine the image and provide the single most likely identification.
            Consider alternatives if appropriate.
            Return ONLY the single most likely identification name (e.g., 'Tomato___Late_blight', 'Aphids', 'Healthy Plant'). Do not add explanation.

            Reference PlantVillage Classes (Examples):
            {', '.join(PLANT_VILLAGE_CLASSES[:15])}... and others.
            """
        else:
            prompt = f"""Analyze the provided image of a plant. Identify the most likely issue:
        1. Is it a plant disease? If yes, identify the disease, preferably using a class name similar to those in the PlantVillage dataset (e.g., 'Tomato___Late_blight', 'Apple___healthy').
        2. Is it a common plant pest? If yes, identify the pest (e.g., 'Aphids', 'Spider Mites').
        3. Is it a common weed? If yes, identify the weed (e.g., 'Dandelion', 'Crabgrass').
        4. If it appears healthy or none of the above apply, state 'Healthy Plant' or 'Unknown/Not Plant'.

        Return ONLY the single most likely identification name (e.g., 'Tomato___Late_blight', 'Aphids', 'Dandelion', 'Healthy Plant', 'Unknown/Not Plant'). Do not add explanation.

        Reference PlantVillage Classes (Examples):
        {', '.join(PLANT_VILLAGE_CLASSES[:15])}... and others.
        """
        response = vision_model.generate_content([prompt, img], stream=False, safety_settings=safety_settings)
        response.resolve()
        identified_issue = response.text.strip()
        if is_reanalysis:
             st.write(f"DEBUG: Gemini Vision RE-ANALYSIS Output for {filename}: {identified_issue}") # DEBUG for re-analysis
        else:
             st.write(f"DEBUG: Gemini Vision Initial Output for {filename}: {identified_issue}") # DEBUG for initial analysis
        analysis_end_time = time.time()
        return {
            "identified_issue": identified_issue,
            "error": None,
            "processing_time": analysis_end_time - analysis_start_time
        }
    except Exception as e:
        error_msg = f"Gemini Vision API call failed for {filename} (Re-analysis: {is_reanalysis}): {e}"
        st.error(error_msg) # Show error in UI
        return {"identified_issue": "Error", "error": error_msg, "processing_time": time.time() - analysis_start_time}


def get_issue_details(issue_name, filename):
    """Calls Gemini Text model to get details about the identified issue."""
    if not issue_name or issue_name in ["Unknown/Not Plant", "Error", "Healthy Plant"]:
        return {"details": "No specific details needed for this classification.", "error": None}

    analysis_start_time = time.time()
    try:
        prompt = f"""Provide detailed information about the plant issue: "{issue_name}".
        Include the following sections if applicable:
        1.  **Type:** (Disease, Pest, Weed)
        2.  **Common Name(s):** (If different)
        3.  **Symptoms/Description:** Describe key visual signs or characteristics.
        4.  **Cause/Biology:** Briefly explain (e.g., fungus, insect lifecycle, weed growth habit).
        5.  **Affected Plants:** List common host plants.
        6.  **Management/Treatment:** Suggest common management strategies. Compare organic and chemical options where relevant, mentioning general pros/cons (e.g., effectiveness, environmental impact, cost considerations if known generally - avoid specific prices).
        7.  **Prevention:** Provide tips to prevent the issue.

        Format the response clearly using Markdown.
        """
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
        error_msg = f"Gemini Text API call failed for details on {issue_name} ({filename}): {e}"
        st.error(error_msg) # Show error in UI
        return {"details": "Error retrieving details.", "error": error_msg, "processing_time": time.time() - analysis_start_time}

# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Choose plant images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or more images of plants for analysis."
)

# --- Display Uploaded Images (Optional Preview) ---
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
            # Ensure forecast_info["data"] is not None before passing
            forecast_list_data = forecast_info.get("data")
            if forecast_list_data is not None:
                 disease_alerts = check_disease_risk(selected_plant, forecast_list_data, DISEASE_RISK_RULES)
            else:
                 st.warning("Forecast data received was empty or invalid.") # Handle case where data is None
                 disease_alerts = []


        if disease_alerts:
            st.warning("Potential Disease Risks Based on Forecast:")
            for alert in disease_alerts:
                st.markdown(f"- {alert}")
        elif not forecast_info["error"]:
            st.success(f"Forecast conditions do not indicate high risk for common diseases for {selected_plant} in {user_city} in the next 48 hours.")
        # Optionally display some forecast summary here if needed
        # st.write("Forecast Data Snippet:", forecast_info["data"][:3]) # Debug: show first few forecast points

elif user_city and selected_plant == "Select Plant...":
    st.info("Select a plant type from the sidebar to get disease risk alerts.")
elif not user_city:
     st.info("Enter a city name in the sidebar to get weather forecasts and disease risk alerts.")


# --- Analysis Trigger ---
st.markdown("---") # Separator
analyze_button = st.button("Analyze Uploaded Images", type="primary", disabled=not uploaded_files)

# --- Session State Initialization ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

# --- Analysis Execution ---
if analyze_button and uploaded_files:
    st.session_state.analysis_results = {} # Clear previous results
    start_time_total = time.time()
    progress_bar = st.progress(0, text="Starting analysis...")

    # Fetch CURRENT weather data once if city is provided (still useful for context)
    weather_info = {"data": None, "error": None}
    if user_city:
        # This re-fetches current weather, could potentially reuse forecast data if needed
        weather_info = get_weather_data(user_city, OPENWEATHER_API_KEY)
        if weather_info["error"]:
            # Don't show warning here again if forecast failed, avoid redundancy
            pass

    with st.spinner("Analyzing images... This may take a moment."):
        num_files = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            image_bytes = uploaded_file.getvalue()
            # Include CURRENT weather info in results for context
            current_result = {"weather": weather_info}

            # Calculate progress percentage
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

            # Store combined result
            st.session_state.analysis_results[filename] = current_result
            progress_bar.progress(max_progress, text=f"Finished: {filename}")
            time.sleep(0.05) # Small pause for UI

    progress_bar.empty()
    total_time = time.time() - start_time_total
    st.success(f"Image analysis complete for {num_files} images in {total_time:.2f} seconds.")

# --- Display Analysis Results ---
st.markdown("---")
st.subheader("📊 Image Analysis Results")

if not st.session_state.analysis_results and not analyze_button: # Show only if button not clicked yet
     st.info("Upload images and click 'Analyze Uploaded Images' to see results.")
elif not st.session_state.analysis_results and analyze_button: # Show if button clicked but no results (e.g., error during analysis)
     st.warning("Analysis was run, but no results were generated. Check for errors above.")
else:
    image_filenames = list(st.session_state.analysis_results.keys())
    tab_titles = [f"{fname[:20]}{'...' if len(fname)>20 else ''}" for fname in image_filenames]
    tabs = st.tabs(tab_titles)

    for i, tab in enumerate(tabs):
        with tab:
            filename = image_filenames[i]
            result = st.session_state.analysis_results[filename]

            col1, col2 = st.columns([1, 2]) # Create columns for layout

            with col1:
                # --- Display Image ---
                # Find the corresponding uploaded file object again to display
                uploaded_file_obj = None
                if uploaded_files: # Check if uploaded_files still exists
                    uploaded_file_obj = next((f for f in uploaded_files if f.name == filename), None)

                if uploaded_file_obj:
                    try:
                        img_display = Image.open(uploaded_file_obj)
                        st.image(img_display, caption=filename, use_container_width=True)
                    except Exception as img_err:
                        st.warning(f"Could not display image {filename}: {img_err}")
                else:
                    st.caption(f"Image data for {filename} not available for display.") # Less alarming message

                # --- Display Blur Warning ---
                blur_info = result.get("blur_info")
                if blur_info and blur_info["is_blurry"]:
                    st.warning(f"⚠️ Possible Blur (Variance: {blur_info['variance']:.2f})")

                # --- Display CURRENT Weather Context ---
                weather = result.get("weather", {}) # This holds CURRENT weather
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
                # --- Display Identification ---
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

                # --- Feedback Section (Moved before Details to trigger re-analysis if needed) ---
                st.divider()
                feedback_options = ["Not Rated", "Accurate", "Inaccurate", "Partially Accurate", "Unsure"]
                feedback_key = f"feedback_{filename}"
                # Get previous feedback to detect change
                previous_feedback = st.session_state.feedback.get(filename, "Not Rated")
                try: current_index = feedback_options.index(previous_feedback)
                except ValueError: current_index = 0

                user_feedback = st.radio(
                    "Rate Analysis Quality:", options=feedback_options, index=current_index,
                    key=feedback_key, horizontal=True # label_visibility="collapsed" # Keep label for clarity
                )

                # --- Re-analysis Logic ---
                reanalysis_result_display = None
                trigger_states = ["Inaccurate", "Partially Accurate", "Unsure"]
                # Check if feedback changed *to* a trigger state
                if user_feedback in trigger_states and user_feedback != previous_feedback:
                    st.session_state.feedback[filename] = user_feedback # Update feedback state first
                    with st.spinner(f"Re-analyzing {filename} based on feedback..."):
                         # Get the original image bytes again
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
                             # Store reanalysis result in session state *within* the main result dict
                             st.session_state.analysis_results[filename]['reanalysis'] = reanalysis_result
                             reanalysis_result_display = reanalysis_result # Prepare to display immediately
                         else:
                             st.warning("Could not perform re-analysis (image data missing or initial ID not suitable).")
                    st.rerun() # Rerun script to update display immediately after re-analysis

                elif user_feedback != previous_feedback:
                     # Feedback changed, but not to a trigger state, just update it
                     st.session_state.feedback[filename] = user_feedback
                     # Clear any previous re-analysis if feedback is now Accurate/Not Rated
                     if 'reanalysis' in st.session_state.analysis_results[filename]:
                         del st.session_state.analysis_results[filename]['reanalysis']
                     # No rerun needed if just changing feedback away from trigger

                # Check if a stored re-analysis result exists for display
                stored_reanalysis = result.get('reanalysis')
                if stored_reanalysis and user_feedback in trigger_states:
                     reanalysis_result_display = stored_reanalysis


                # --- Display Re-analysis Result (if available) ---
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
                    # Optionally, add a button to get details for the re-analyzed issue?

                # --- Display Details (Now uses original identified_issue) ---
                details_info = result.get("details", {})
                # Only fetch/display details for the *original* identification for now
                # Could be enhanced later to fetch details for re-analysis result if needed
                details_text = details_info.get("details", "N/A")
                details_error = details_info.get("error")
                details_time = details_info.get("processing_time", 0)

                st.markdown(f"**Details & Management (for {identified_issue}):**") # Clarify which ID details are for
                if details_error:
                    st.error(f"Details Error: {details_error}")
                else:
                    st.markdown(details_text) # Display Markdown details
                if details_time > 0:
                    st.caption(f"Details Time: {details_time:.2f}s")

                # Feedback radio moved up to trigger re-analysis logic


# --- Sidebar Bottom ---
st.sidebar.divider()
st.sidebar.header("ℹ️ Info & Feedback")

# --- Session Feedback Summary ---
st.sidebar.subheader("📝 Session Feedback")
feedback_summary = [{"Image": fname, "Feedback": fback} for fname, fback in st.session_state.feedback.items() if fback != "Not Rated"]
if feedback_summary:
    st.sidebar.dataframe(feedback_summary, use_container_width=True)
else:
    st.sidebar.caption("No feedback provided yet.")

st.sidebar.divider()

# --- How to Use ---
with st.sidebar.expander("How to Use"):
    st.markdown("""
    1.  **API Keys:** Ensure `GOOGLE_API_KEY` and `OPENWEATHER_API_KEY` are in `.streamlit/secrets.toml`.
    2.  **Enter City:** Provide your city name for weather context and alerts.
    3.  **Select Plant:** Choose a plant type for disease risk alerts.
    4.  **Upload Images:** Upload one or more plant images for analysis.
    5.  **Analyze:** Click 'Analyze Uploaded Images'.
    6.  **View Results:** Check the main area for alerts and the tabs for image-specific analysis.
    7.  **Provide Feedback:** Rate the analysis quality using the radio buttons under each image result.
    """)

st.sidebar.divider()
st.sidebar.caption("Powered by Google Gemini & OpenWeatherMap.")

# Command to run: python3 -m streamlit run app.py
