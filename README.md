# FitSync Pro Backend

## Setup

1. Clone the repo and enter the backend folder.
2. Copy `.env.example` to `.env` and configure variables.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the server:
   ```
   uvicorn main:app --reload
   ```
5. For Docker:
   ```
   docker build -t fitsync-backend .
   docker run -p 8000:8000 fitsync-backend
   ```

## API Endpoints

- `/register` - Register a new user
- More endpoints to be added...

## Tech Stack

- FastAPI, SQLAlchemy, Pydantic, Docker, etc.
