from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence, List
from pydantic import BaseModel , Field

from langgraph.graph.message import add_messages

class SearchTerms(BaseModel):
    """Schema for search terms generation"""
    search_terms: List[str] = Field(...,
        description="List of diverse search terms for comprehensive information gathering"
    )

parser = PydanticOutputParser(pydantic_object=SearchTerms)

class AttractionPlannerState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    travel_type: str
    currency : str
    attraction_result: str

class HotelPlannerState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    budget: str
    travel_type: str
    currency : str
    hotel_result: str

class FoodPlacesPlanner(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    travel_type: str
    budget : str
    currency : str
    restaurant_result: str

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    attractions: AttractionPlannerState
    hotels: HotelPlannerState
    restaurants: FoodPlacesPlanner
    currency: str
    travel_type: str
    final_result: str
    
