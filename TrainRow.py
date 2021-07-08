from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class TrainRow(BaseModel):
    transaction_description: str
    submerchant: str
