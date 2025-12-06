from typing import Dict

class Calibrator:
    """
    Adjusts raw confidence scores based on historical performance.
    """
    
    def __init__(self):
        # Heuristic calibration map (Sector -> Win Rate Adjustment)
        # In the future, this will be learned from data
        self.sector_adjustments = {
            "Technology": 1.0,   # Neutral
            "Finance": 0.95,     # Slightly harder to predict
            "Healthcare": 0.90,  # Harder
            "Energy": 1.05,      # Easier (trend driven)
            "Unknown": 0.90
        }

    def calibrate_score(self, raw_score: float, sector: str = "Unknown") -> float:
        """
        Adjusts the raw score based on sector-specific calibration.
        
        Args:
            raw_score: The raw confidence score (0-100).
            sector: The sector of the stock.
            
        Returns:
            Calibrated score (0-100).
        """
        adjustment = self.sector_adjustments.get(sector, 0.90)
        calibrated = raw_score * adjustment
        return min(100.0, max(0.0, calibrated))
