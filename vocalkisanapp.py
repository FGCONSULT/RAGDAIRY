import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# 1. Initialize environment variables from a local .env file (if present)
load_dotenv()

app = Flask(__name__)

# 2. Enable CORS so your secure Wix backend can route requests to this server
CORS(app)

# 3. Retrieve the master security key injected by your hosting environment or .env file
MASTER_API_KEY = os.environ.get('DAIRY_API_KEY')

# Quick server-side safety check log
if not MASTER_API_KEY:
    print("⚠️ WARNING: 'DAIRY_API_KEY' environment variable is not set. Security gatekeeper will fail.")
else:
    print("✅ Secure Gateway Active: 'DAIRY_API_KEY' loaded successfully.")

@app.route('/api/simulate', methods=['POST'])
def simulate_dairy_data():
    # 4. Security Gatekeeper: Inspect the custom incoming header
    client_api_key = request.headers.get('X-API-KEY')

    if not client_api_key or client_api_key != MASTER_API_KEY:
        return jsonify({
            "status": "error",
            "message": "Unauthorized access attempt. Invalid or missing token verification."
        }), 401

    # 5. Core Processing Engine (Executes only if the key is valid)
    payload = request.json or {}
    
    focal_domain = payload.get('focal_domain', 'Milk supply estimation by 2050 and self-sufficiency')
    geospatial = payload.get('geospatial_demarcation', 'Country (National Level View)')
    command = payload.get('command', '').lower().strip()

    # Timeline vector segments (2024 to 2050)
    years = [2024, 2030, 2037, 2045, 2050]
    
    # Base simulation matrix arrays (Million Tonnes / Year)
    bau_trend = [230, 250, 280, 310, 330]
    productivity_boost = [230, 270, 320, 380, 420]
    value_chain_exp = [230, 260, 305, 360, 400]
    climate_resilient = [230, 285, 350, 425, 480]

    # Default dynamic narrative elements
    explanatory_text = (
        "The targeted Climate-Resilient optimization track safely matches domestic consumer "
        "demand indices completely by 2039, ensuring full supply-side self-sufficiency."
    )
    forecast_insight = (
        "Upgrading minor chilling centers to smart automated cold hubs prevents up to "
        "14% systemic wastage under maximum summer temperature profiles."
    )
    
    # Proactive baseline warning tracker
    alert_payload = {
        "active": True,
        "status": "warning",
        "title": "Processing Infrastructure Strain",
        "message": "Fodder supply volatility models project a local distribution deficit around year 2035 under the BAU pathway."
    }

    # 6. Interactive Dialogue Command Engine Override
    if "deploy adaptive rerouting" in command or "reroute" in command:
        alert_payload = {
            "active": True,
            "status": "success",
            "title": "Adaptive Rerouting Implemented",
            "message": "Fodder logistics redirected. Supply stabilization matrix fully resolved for 2035 deficit vectors."
        }
        explanatory_text += " [Update: AI rerouting mitigation protocols currently operational.]"
        
    elif "override forecast" in command:
        alert_payload = {
            "active": False,
            "status": "neutral",
            "title": "Alert Cleared",
            "message": "System running on user-defined forecast parameters."
        }
        # Compute alternative aggressive development curve vector
        productivity_boost = [230, 290, 360, 430, 500]

    # 7. Formulate structural data payload map
    response_data = {
        "meta": {
            "focal_domain": focal_domain,
            "geospatial_demarcation": geospatial
        },
        "timeline": years,
        "metrics": {
            "BAU": bau_trend,
            "Productivity Boost": productivity_boost,
            "Value Chain Exp": value_chain_exp,
            "Climate Resilient": climate_resilient
        },
        "insights": {
            "explanatory_matrix": explanatory_text,
            "forecast_insight": forecast_insight
        },
        "alert": alert_payload
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    # Run the microservice engine locally on port 5000
    app.run(debug=True, port=5000)
