from flask import Flask, request, jsonify
import jwt, os, json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# load credential
SECRET_KEY = os.getenv('SECRET_KEY')
USERS = os.getenv('USERS')
ALGORITHM = os.getenv('ALGORITHM')

@app.route('/auth/login', methods=['POST'])
def login() -> dict:
    """Endpoint for user authentication."""
    username = request.json.get('username')
    password = request.json.get('password')
    USERS_DB = json.loads(USERS)
    
    if username in USERS_DB and USERS_DB[username]==password:
        access_token = generate_access_token(username)
        return jsonify({'access_token': access_token}), 200
    
    return jsonify({'message': 'Invalid username and password'}), 401

def generate_access_token(username:str) -> str:
    """Helper function to generate a jwt token."""
    payload = {'username': username}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

@app.route('/auth/validate/<access_token>', methods=["GET"])
def validate_access_token(access_token:str) -> bool:
    """Function to validate access token"""
    try:
        jwt.decode(access_token, key=SECRET_KEY, algorithms=[ALGORITHM])
        return jsonify({'token_valid': True}), 200
    except jwt.exceptions.InvalidTokenError:
        return jsonify({'token_valid': False}), 401
    
if __name__ == '__main__':
    app.run(debug=True)