from pydantic import BaseModel, Field, condecimal
from typing import Optional
from datetime import datetime

Decimal18_8 = condecimal(max_digits=18, decimal_places=8)

class Bar1m(BaseModel):
    ts: datetime
    symbol: str = Field(min_length=1)
    open: Decimal18_8
    high: Decimal18_8
    low: Decimal18_8
    close: Decimal18_8
    volume: int
    vwap: Optional[Decimal18_8] = None
    trades_count: Optional[int] = None

    def to_row(self) -> tuple:
        return (
            self.ts, self.symbol, self.open, self.high, self.low,
            self.close, self.volume, self.vwap, self.trades_count
        )
