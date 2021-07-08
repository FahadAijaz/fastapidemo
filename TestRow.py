from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class TestRow(BaseModel):
    transaction_description: str
