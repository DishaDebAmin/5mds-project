from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import random

app = FastAPI()

# Enable CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")  # Health-check or home
def read_root():
    return {"message": "Backend is running. Use POST /generate."}

@app.post("/generate")
async def generate_campaign(
    campaign_name: str = Form(...),
    brand_color: str = Form(...),
    logo: Optional[UploadFile] = File(None)
):
    ad_text = f"Introducing {campaign_name}! The best choice for your style."
    image_url = "https://via.placeholder.com/150.png?text=Brand+Logo"
    suggested_time = random.choice([
        "Tomorrow at 7 PM",
        "Saturday at 3 PM",
        "Friday at 10 AM"
    ])
    return {
        "ad_copy": ad_text,
        "image_url": image_url,
        "suggested_posting_time": suggested_time
    }

# Standard FastAPI entry point if running this script directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
