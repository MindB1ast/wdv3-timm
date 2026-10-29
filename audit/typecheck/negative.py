"""Exactly three intentional consumer errors, checked separately."""
from Scripts.types import WDTagResult


def invalid_consumer(result: WDTagResult) -> None:
    print(result["missing_field"])
    result["general"]["eyes"] = "yellow"
    print(result["provenance"]["eyes"]["source"])
