# plant-assistant
Enhance your README.md with clear instructions. Here’s a template:

🌱 Kisan Krishi Dost AI - Plant Health Assistant
A Streamlit app that uses Google Gemini and OpenWeatherMap to analyze plant images, detect diseases/pests, and provide weather-based risk alerts.

Features
Image Analysis : Identify plant diseases, pests, or weeds using AI.
Weather Context : Show real-time weather data for your location.
Disease Risk Alerts : Predict disease risks based on weather forecasts.
Detailed Information : Get management strategies and prevention tips.
Prerequisites
API Keys :
Google Gemini API Key (for vision/text analysis).
OpenWeatherMap API Key (for weather data).
Setup
Clone the repository:
bash
Copy
1
2
git clone https://github.com/ranchopanda/kisan-krishi-dost-ai.git
cd kisan-krishi-dost-ai
Install dependencies:
bash
Copy
1
pip install -r requirements.txt
Create API keys file:
bash
Copy
1
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
Replace placeholders in .streamlit/secrets.toml with your actual keys.
Run the app:
bash
Copy
1
streamlit run app.py
Usage
Enter a city name in the sidebar for weather data.
Select a plant type (e.g., Tomato, Apple) to get disease risk alerts.
Upload plant images and click "Analyze Uploaded Images" .
Review results in the UI, including blur warnings and detailed analysis.
Screenshots

