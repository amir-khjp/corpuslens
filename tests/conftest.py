import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "The quarterly revenue report shows strong growth across all business units with continued momentum and solid execution this period in Q3.",
                "The quarterly revenue report shows strong growth across all business units with continued momentum and solid execution this period in Q3.",  # exact dup
                "The quarterly revenue report shows strong growth across all business units with continued momentum and solid execution this period in Q4.",  # near dup (one word differs)
                "Our new product launch received excellent customer feedback.",
                "Customers loved the new product launch and gave great feedback.",
                "Contact support at help@example.com or call 555-123-4567.",
                "The weather today is sunny with a light breeze.",
                "It's a sunny day with pleasant breezy weather.",
                "",
                "Short.",
            ],
            "category": ["finance", "finance", "finance", "product", "product", "support", "weather", "weather", "misc", "misc"],
        }
    )
