import operator
from pydantic import BaseModel , Field
from typing import TypedDict, Annotated, Sequence, Literal, Optional
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_question: Optional[str]
    last_node: Optional[str]  # Track which node produced the last output
    retry_count: Optional[int]  # Prevent infinite loops
    validation_feedback: Optional[str]  # Store validation feedback for context
    validation_status: Optional[str]  # Track validation status


class TopicClassification(BaseModel):
    topic: Literal["Data Structures", "General Query", "Web Search"] = Field(
        ...,
        description=(
            "Classify the user query into one of the three categories: Data Structures, General Query, Web Search"
            )
    )
    query : str = Field(..., description = "Original query provided by the user")
    reasoning: str = Field(..., description="Explanation of why the topic was classified this way."
)
    
parser = PydanticOutputParser(pydantic_object = TopicClassification)


class ResponseValidator(BaseModel):
    is_valid : bool = Field(..., description = "Whether the response was valid or not")
    feedback : str = Field(..., description = "Feedback provided by the LLM")

valid_parser = PydanticOutputParser(pydantic_object = ResponseValidator)    