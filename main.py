from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
# import connectToDatabase
import requests
import httpx
import pandas as pd
from pandas import json_normalize
import pandas as pd
import numpy as np
import torch
from torch import nn
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class NCFModel(nn.Module):
    def __init__(self, num_users, num_events, embedding_dim):
        super(NCFModel, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.event_emb = nn.Embedding(num_events, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, user_ids, event_ids):
        user_vecs = self.user_emb(user_ids)
        event_vecs = self.event_emb(event_ids)
        x = torch.cat([user_vecs, event_vecs], dim=1)
        return self.mlp(x).squeeze(1)

app = FastAPI()

class UserRequest(BaseModel):
    userId: str

users_new_df = None
events_train_df = None
user_map = {}
event_map = {}
num_users = 0
num_events = 0
embedding_dim = 16
model = None

async def fetch_data(url: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()  # Ensure we catch errors
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error {e.response.status_code}: {e}"}
        except httpx.RequestError as e:
            return {"error": f"Request error: {e}"}


async def prepare_data():
    global users_new_df, events_train_df, user_map, event_map, num_users, num_events, model

    user_url = "http://localhost:8080/users/user-preference/list"
    event_url = "http://localhost:8080/users/event-preference/list"

    user_data = await fetch_data(user_url)
    event_data = await fetch_data(event_url)

    # User
    users=json_normalize(user_data)
    data_users=users.iloc[0]
    users_df = pd.DataFrame(data_users)
    new_data = []
    for i in range(len(users_df[0]['userId'])):
        userId = users_df[0]['userId'][i]
        categories = users_df[0]['categories'][i]

        new_data.append({
            "userId": userId,
            "categories": categories
        })
    users_new_df = pd.DataFrame(new_data)
    print(users_new_df)

    # Event
    events=json_normalize(event_data)
    data_events=events.iloc[0]
    events_df = pd.DataFrame(data_events)
    new_events_data = []
    for i in range(len(events_df[0]['id'])):
        Id = events_df[0]['id'][i]
        name = events_df[0]['name'][i]
        categories = events_df[0]['categories'][i]

        new_events_data.append({
            "id": Id,
            "name": name,
            "categories": categories
        })
    events_train_df = pd.DataFrame(new_events_data)
    print(events_train_df)

    user_map = {uid: idx for idx, uid in enumerate(users_new_df["userId"])}
    event_map = {eid: idx for idx, eid in enumerate(events_train_df["id"])}

    num_users = len(user_map)
    num_events = len(event_map)

    # Load model
    model = NCFModel(num_users, num_events, embedding_dim)
    model.load_state_dict(torch.load("ncf_model.pth"))
    model.eval()

    print(model)

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet")

@app.on_event("startup")
async def on_startup():
    await prepare_data()

@app.post("/recommendations/")
async def get_recommendations(user: UserRequest):
    try:
        user_idx = user_map.get(user.userId)
        if user_idx is None:
            raise HTTPException(status_code=404, detail="User not found")
        
        with torch.no_grad():
            user_ids = torch.tensor([user_idx] * num_events, dtype=torch.long)
            event_ids = torch.tensor(range(num_events), dtype=torch.long)
            scores = model(user_ids, event_ids)

        recommended_events = events_train_df.copy()
        print(f"Scores shape: {scores.shape}")
        print(f"Recommended events shape: {recommended_events.shape}")
        recommended_events["score"] = scores.numpy()

        recommended_events = recommended_events.sort_values(by="score", ascending=False)
        return recommended_events.head(5)[["id", "name", "score"]].to_dict(orient="records")

    except Exception as e:
        logger.exception("Error in recommendation endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Event Recommendation API is running!"}

# if __name__ == "__main__":
#     conn = connectToDatabase.get_connection()
#     cur = conn.cursor()

#     cur.execute("SELECT version();")
#     db_version = cur.fetchone()
#     print(f"Connected to database version: {db_version}")