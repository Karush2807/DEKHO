from fastapi import FastAPI
from api.tracking import router as tracking_router

app = FastAPI()

# Include tracking endpoints
app.include_router(tracking_router)

@app.get("/")
def home():
    return {"message": "FastAPI Backend Running!"}
