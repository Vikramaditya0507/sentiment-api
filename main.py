import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Create FastAPI app
app = FastAPI()

# Initialize OpenAI client using environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request model
class CommentRequest(BaseModel):
    comment: str

# Response model
class CommentResponse(BaseModel):
    sentiment: str
    rating: int

# POST /comment endpoint
@app.post("/comment", response_model=CommentResponse)
def analyze_comment(req: CommentRequest):
    try:
        # Validate input
        if not req.comment.strip():
            raise HTTPException(status_code=400, detail="Comment cannot be empty")

        # Call OpenAI with structured output
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Analyze sentiment of this comment: {req.comment}",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"]
                    }
                }
            }
        )

        data = response.output[0].content[0].parsed

        return {
            "sentiment": data["sentiment"],
            "rating": data["rating"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))