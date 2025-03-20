# Talent-Atmos-Rec

This repository is used for developing the Neural Collaborative Filtering model using deep learning with <br />
the general matrix factorization to enhanced the feature extraction from events data and user data.

## Dependencies
There are the package that require for this project in the requirement.txt file.
```
pip install -r requirement.txt
```

## Run the virtual Python Environment
If you using Windows for activating venv, run this command:
```
.\venv\Scripts\activate
```
Activate venv for macOS/Linux
```
source \venv\Scripts\activate
```

## Start the app
For Starting the FastAPI application run this command.
```
uvicorn main:app --host localhost --port 7000 --reload --log-level debug
```
