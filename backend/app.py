from flask import Flask, send_from_directory, jsonify, request, redirect
from flask_cors import CORS
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from model_engine import FightPredictor
from research import OddsFeed, ResearchData
from forecast import ForecastEngine

app = Flask(__name__, static_folder=os.path.join(_root, 'public'))
CORS(app)

print("Initializing ML Engine...")
predictor = FightPredictor()
_data_dir = os.path.join(_root, 'data')
_data_path = os.path.join(_data_dir, 'ufc-master.csv')
_v2_path = os.path.join(_data_dir, 'model_v2.pkl')
_legacy_path = os.path.join(_data_dir, 'model.pkl')
_artifact_path = _v2_path if os.path.exists(_v2_path) else _legacy_path

if os.path.exists(_artifact_path):
    print(f"Loading pre-trained model from {_artifact_path}...")
    success = predictor.load_artifact(_artifact_path, _data_path)
else:
    success = False

if not success:
    print("WARNING: Model failed to initialize. Make sure data is present.")

_runtime_dir = os.path.join(_data_dir, 'evaluation', 'identity-repair-v1')
research_data = ResearchData(
    predictor, _artifact_path,
    bundle_directory=os.path.join(_runtime_dir, 'research-data'),
    career_directory=os.path.join(_data_dir, 'evaluation', 'career-supplement-v1'),
    career_manifest_sha256='40c62cd008b45b8d81ddb2c39adcd5732e8fea3459b3a52296be6938a314718e',
)
odds_feed = OddsFeed()
forecast_engine = ForecastEngine(
    os.path.join(_runtime_dir, 'models', 'provisional_challenger.joblib'),
    os.path.join(_runtime_dir, 'candidate_contract.json'),
    research_data,
    os.path.join(_runtime_dir, 'inference', 'fighters.parquet'),
)


def safe(val):
    if val is None or (isinstance(val, (float, np.floating)) and not np.isfinite(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def build_profile(stats):
    wins = safe(stats.get('Wins'))
    losses = safe(stats.get('Losses'))
    draws = safe(stats.get('Draws'))
    ko_wins = [safe(stats.get(key)) for key in ('WinsByKO', 'WinsByTKODoctorStoppage')]
    dec_wins = [safe(stats.get(key)) for key in ('WinsByDecisionUnanimous', 'WinsByDecisionSplit', 'WinsByDecisionMajority')]

    height_cm = safe(stats.get('HeightCms'))
    reach_cm = safe(stats.get('ReachCms'))

    return {
        "available": stats.get('ProfileAsOf') is not None,
        "asOf": stats.get('ProfileAsOf'),
        "featureAsOf": stats.get('FeatureAsOf'),
        "record": f"{wins:g}-{losses:g}-{draws:g}" if all(v is not None for v in (wins, losses, draws)) else None,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "koWins": sum(ko_wins) if all(v is not None for v in ko_wins) else None,
        "subWins": safe(stats.get('WinsBySubmission')),
        "decWins": sum(dec_wins) if all(v is not None for v in dec_wins) else None,
        "avgSigStrLanded": safe(stats.get('AvgSigStrLanded')),
        "avgSigStrPct": safe(stats.get('AvgSigStrPct')),
        "avgTDLanded": safe(stats.get('AvgTDLanded')),
        "avgTDPct": safe(stats.get('AvgTDPct')),
        "avgSubAtt": safe(stats.get('AvgSubAtt')),
        "height": f"{round(height_cm / 2.54)}\u2033" if height_cm else None,
        "heightCm": height_cm,
        "reach": f"{round(reach_cm / 2.54)}\u2033" if reach_cm else None,
        "reachCm": reach_cm,
        "weight": safe(stats.get('WeightLbs')),
        "age": safe(stats.get('Age')),
        "stance": stats.get('Stance') if isinstance(stats.get('Stance'), str) else None,
        "currentWinStreak": safe(stats.get('CurrentWinStreak')),
        "currentLoseStreak": safe(stats.get('CurrentLoseStreak')),
        "longestWinStreak": safe(stats.get('LongestWinStreak')),
        "totalRounds": safe(stats.get('TotalRoundsFought')),
        "titleBouts": safe(stats.get('TotalTitleBouts')),
        "totalFights": safe(stats.get('TotalFights')),
    }


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    pages = {'research': 'predict.html', 'results': 'results.html', 'methodology': 'about.html'}
    legacy = {'index.html': '/', 'predict.html': '/research', 'about.html': '/methodology', 'results.html': '/results'}
    destination = legacy.get(path)
    if path.endswith('/') and path.rstrip('/') in pages:
        destination = f'/{path.rstrip("/")}'
    if destination:
        query = request.query_string.decode('utf-8')
        return redirect(f'{destination}?{query}' if query else destination, code=308)
    return send_from_directory(app.static_folder, pages.get(path, path))


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Endpoint not found"}), 404
    return send_from_directory(app.static_folder, '404.html'), 404


@app.route('/api/fighters', methods=['GET'])
def get_fighters():
    return jsonify({"fighters": research_data.get_fighter_names()})


@app.route('/api/recent-fights', methods=['GET'])
def recent_fights():
    if research_data.research_fights is None:
        return jsonify({"error": "Recent fight records are unavailable"}), 503
    return jsonify(research_data.recent_fights())


@app.route('/api/evaluation', methods=['GET'])
@app.route('/evaluation.json', methods=['GET'])
def evaluation():
    if not forecast_engine.available:
        return jsonify({"error": "The corrected model evaluation is unavailable"}), 503
    try:
        content = (Path(_runtime_dir) / 'evaluation.json').read_bytes()
    except OSError:
        return jsonify({"error": "The corrected model evaluation is unavailable"}), 503
    if hashlib.sha256(content).hexdigest() != forecast_engine.model_spec.get('evaluationSha256'):
        return jsonify({"error": "The evaluation does not match the current model"}), 503
    response = jsonify(json.loads(content))
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify(predictor.get_model_info())


@app.route('/api/research-status', methods=['GET'])
def research_status():
    return jsonify(research_data.status())


@app.route('/api/fighter-history', methods=['GET'])
def fighter_history():
    name = request.args.get('name', '').strip()
    if not name:
        return jsonify({"error": "A fighter name is required"}), 400
    history = research_data.history(name)
    if history is None:
        return jsonify({"error": "Fighter not available in this dataset"}), 404
    if research_data.research_fights is None:
        return jsonify({"error": "Fight history is unavailable"}), 503
    return jsonify(history)


@app.route('/api/matchup-research', methods=['GET'])
def matchup_research():
    if research_data.research_fights is None:
        return jsonify({"error": "Matchup research is unavailable"}), 503
    try:
        return jsonify(research_data.matchup_evidence(request.args.get('fighter1', ''), request.args.get('fighter2', '')))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route('/api/odds', methods=['GET'])
def odds():
    payload, status = odds_feed.fetch()
    return jsonify(payload), status


@app.route('/api/forecast-status', methods=['GET'])
def forecast_status():
    return jsonify(forecast_engine.status())


@app.route('/api/forecast', methods=['POST'])
def forecast():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request must be a JSON object"}), 400
    fields = {'firstID', 'secondID', 'eventDate', 'weightClass', 'rounds', 'titleBout'}
    if set(data) != fields:
        return jsonify({"error": "Provide both fighter identities, fight date, division, rounds and title-bout status"}), 400
    if not forecast_engine.available:
        return jsonify({"error": forecast_engine.error}), 503
    try:
        return jsonify(forecast_engine.predict(
            data['firstID'], data['secondID'], data['eventDate'],
            data['weightClass'], data['rounds'], data['titleBout'],
        ))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    return jsonify({"error": "The original snapshot model has been retired because its fighter histories contain identity errors. Use the matchup estimate above."}), 410


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=2102)
