from typing import Optional

class EnrichmentService:
    """Mock service to resolve hard anchors into human-readable descriptions."""

    _MOCK_DB = {
        "JIRA-123": "Critical Sharding Failure in Database Layer",
        "PR-45": "Refactor: Move segmentation logic to shared service",
        "GH-#5030": "Feat: Add attach limit autoscaler",
        "KEP-1234": "KEP: Horizontal Pod Autoscaling v2"
    }

    def resolve_anchor(self, anchor_type: str, anchor_value: str) -> Optional[str]:
        """Resolves an anchor value to a description.
        
        Args:
            anchor_type: The type of anchor (e.g., TICKET, PR).
            anchor_value: The value of the anchor (e.g., JIRA-123).
            
        Returns:
            Optional[str]: The description if found, else None.
        """
        if not anchor_value:
            return None
            
        normalized_value = anchor_value.strip()
        return self._MOCK_DB.get(normalized_value)
