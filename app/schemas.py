"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input features for house price prediction.

    Features are based on the California Housing dataset:
    - MedInc: Median income in block group (in tens of thousands of dollars)
    - HouseAge: Median house age in block group (years)
    - AveRooms: Average number of rooms per household
    - AveBedrms: Average number of bedrooms per household
    - Population: Block group population
    - AveOccup: Average number of household members
    - Latitude: Block group latitude
    - Longitude: Block group longitude
    """

    MedInc: float = Field(..., gt=0, description="Median income (tens of thousands USD)")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age (years)")
    AveRooms: float = Field(..., gt=0, description="Average rooms per household")
    AveBedrms: float = Field(..., gt=0, description="Average bedrooms per household")
    Population: float = Field(..., gt=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Average household members")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 3.5,
                    "HouseAge": 25.0,
                    "AveRooms": 5.0,
                    "AveBedrms": 1.0,
                    "Population": 1500.0,
                    "AveOccup": 3.0,
                    "Latitude": 37.5,
                    "Longitude": -122.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Model prediction output."""

    predicted_price: float = Field(
        ..., description="Predicted median house value (hundreds of thousands USD)"
    )
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service health status")


class ModelInfoResponse(BaseModel):
    """Information about the loaded ML model."""

    model_name: str
    model_version: str
    features: list[str]
    description: str


class APIInfoResponse(BaseModel):
    """Basic API information."""

    name: str
    version: str
    description: str
    docs_url: str
