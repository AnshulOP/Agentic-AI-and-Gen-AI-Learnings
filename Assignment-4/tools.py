from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from typing import List

from dotenv import load_dotenv
from schemas import SearchTerms

import requests
import json
import os

load_dotenv()

llm = ChatOpenAI(model = "o4-mini-2025-04-16")
parser = PydanticOutputParser(pydantic_object = SearchTerms)
        
@tool
def get_weather_data(city: str):
    """Function to fetch the current weather details for a given city"""
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    return response.json()

@tool
def web_search_general(query: str):
    """Function that generates diverse and effective search terms from a user query and performs web search for each."""

    prompt = PromptTemplate(
        template="""
        You are a smart, research-oriented assistant whose task is to generate **high-quality, diverse, and well-targeted search phrases** based on the user's input query.

        Your output will guide a web search system to gather the **most relevant, accurate, and well-rounded information**, helping a language model curate insightful content.

        ---
        🎯 **OBJECTIVE**
        Analyze the user query from all angles and generate **10 to 12 distinct search terms** that:
        - Explore **different subtopics, perspectives, or intent types** (e.g. informational, practical, comparative, solution-oriented)
        - Capture **key concepts and possible user goals**
        - Are optimized for **maximum discoverability** of accurate and up-to-date web content

        ---
        🧠 **SEARCH TERM GENERATION PRINCIPLES**
        - Each search phrase should be **5 to 12 words long** and focused.
        - Cover **different dimensions or facets** of the user query (not just reworded variations).
        - Use a mix of **styles** to capture varied context.
        - Include **contextual or situational keywords** where applicable.
        - Avoid generic repetition — ensure each term adds a **distinct direction** to the search.
        - Adapt and be specific to the **domain of the user query** (e.g. travel, exploration, science, shopping, etc.)

        ---
        **User Query**: {query}

        Format Instructions: {format_instructions}
        """,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "query": query
    })

    tavily_tool = TavilySearch(max_results=3, include_answer = True)

    search_queries = response.search_terms

    query_results = {}
    for search_term in search_queries:
        result = tavily_tool.invoke(search_term)
        query_results[search_term] = result['answer']

    return json.dumps(query_results)

@tool
def web_search_food_places(query: str):
    """Function that generates diverse and effective search terms from a user query and performs web search for each."""

    prompt = PromptTemplate(
        template="""
        You are a smart and versatile assistant helping to plan memorable food experiences for travelers.

        Your job is to **generate 10 to 12 high-quality search terms** based on a user's query about food, restaurants, cafes, or local cuisines.

        These search terms will be used to find diverse and up-to-date information from the web about the **best places to eat, drink, and explore culinary culture** in the given location.

        ---
        🎯 **INSTRUCTION**
        - Generate search phrases that explore **different facets of the food experience**, such as:
        - Specific mealtimes (breakfast, lunch, snacks, dinner)
        - Cuisine types (local, fusion, international)
        - Dining styles (street food, rooftop cafes, fine dining)
        - Nearby landmarks or stay proximity
        - Think like a **food blogger + local expert** optimizing for web search.
        - Tailor each phrase for **precision and coverage** — avoid redundancy.

        ---
        🧠 **GUIDELINES**
        - Each search phrase should be **5 to 15 words long**
        - Adapt and be specific to the **domain of the user query** (include what is stricitly required as per query which can influence the search results)
        - Use diverse styles like:
            - “Best rooftop restaurants for dinner in [City]”
            - “Local street food to try [City]”
            - “Top cafes for brunch in [City] with reviews”
            - Ensure **each term is distinct** and covers a unique angle (e.g., time, style, vibe)
            - Optimize for tools that depend on precise search queries (e.g., Tavily, SerpAPI)

        ---
        **User Query**: {query}

        Format Instructions: {format_instructions}
        """,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
    )


    chain = prompt | llm | parser

    response = chain.invoke({
        "query": query
    })

    tavily_tool = TavilySearch(max_results=3, include_answer = True)

    search_queries = response.search_terms

    query_results = {}
    for search_term in search_queries:
        result = tavily_tool.invoke(search_term)
        query_results[search_term] = result['answer']

    return json.dumps(query_results)

@tool
def calculate_total_expense(costs: List[float]) -> float:
    """Calculate total expense from a list of costs"""
    return sum(costs)

@tool
def get_conversion_rate(base_currency: str, target_currency: str) -> float:
    """Get the conversion rate between two currencies.
    
    Args:
        base_currency: The base currency code (e.g., USD)
        target_currency: The target currency code (e.g., INR)"""
    
    url = f"https://v6.exchangerate-api.com/v6/92e81466cf66691552fa8d6e/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    data = response.json()
    
    return data['conversion_rate']

@tool
def convert(base_currency_values: List[float], conversion_rate: float) -> List[float]:
    """
    Convert a list of amounts using a conversion rate.

    Args:
        base_currency_values: A list of amounts in base currency
        conversion_rate: The conversion rate from base to target currency

    Returns:
        A list of converted amounts, each rounded to 2 decimal places
    """
    return [round(value * conversion_rate, 2) for value in base_currency_values]

@tool
def multiply_numbers(x: float, y: float) -> float:
    """
    Multiplies two numbers and returns the result.
    
    Args:
        x: The first number.
        y: The second number.
    
    Returns:
        The product of the two numbers.
    """
    return x * y
