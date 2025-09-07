import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
CORS(app)

# Simulated in-memory storage for demonstration
users = {}
workouts = {}

@app.route('/api/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'GET':
        user_id = request.args.get('user_id', 'default')
        return jsonify(users.get(user_id, {})), 200
    elif request.method == 'POST':
        data = request.json
        user_id = data.get('user_id', 'default')
        users[user_id] = {
            "name": data.get("name", ""),
            "email": data.get("email", "")
        }
        return jsonify({"message": "Profile updated"}), 200

@app.route('/api/workouts', methods=['GET', 'POST'])
def workout():
    user_id = request.args.get('user_id', 'default') if request.method == 'GET' else request.json.get('user_id', 'default')
    if request.method == 'GET':
        return jsonify(workouts.get(user_id, [])), 200
    elif request.method == 'POST':
        data = request.json
        entry = {
            "workout": data.get("workout", ""),
            "duration": data.get("duration", 0),
            "notes": data.get("notes", "")
        }
        if user_id not in workouts:
            workouts[user_id] = []
        workouts[user_id].append(entry)
        return jsonify({"message": "Workout added"}), 200

@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
        description = e.description
    else:
        description = str(e)
    return jsonify({"error": description}), code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)