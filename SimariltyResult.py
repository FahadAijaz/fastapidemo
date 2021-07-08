from pydantic import BaseModel
from typing import List

class SimilarityScore(BaseModel):
    similarity_score: float
    similarity_score_index: int
    test_transaction_description: str
    train_transaction_description: str
    predicted_submerchant: str

class SimilarityScoreCollection(BaseModel):
    similarity_score_collection: List[SimilarityScore]