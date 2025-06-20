
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

from schemas import AttractionPlannerState
from tools import get_weather_data, get_conversion_rate, web_search_general, calculate_total_expense, convert, multiply_numbers, web_search_food_places

class Attractions:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(model = "o4-mini-2025-04-16")
        self.required_tools = [get_weather_data, web_search_general, calculate_total_expense, get_conversion_rate, multiply_numbers, convert]
        self.llm_with_tools = self.llm.bind_tools(self.required_tools)

    def llm_attraction_call(self, state : AttractionPlannerState):

        question = state["messages"]
        print(question)

        prompt = PromptTemplate(
        template="""
            You are a smart and helpful travel assistant.

            Your task is to generate a detailed, accurate, and practical travel itinerary for the user based on the days provided in the user query and any additional available information such as recent search results, current weather, local transport options, ticket prices, and expert travel tips.

            Think like a professional travel planner: optimize for weather suitability, logistics, user convenience, budget, and experience variety. Use the most recent data available from the web whenever required, especially for transportation or pricing details.


            📚 Available Information Includes (but not limited to):
            - Popular and recent attractions
            - Activity suggestions
            - Current weather and seasonal insights
            - Entry timings, ticket prices, and schedules
            - Public and private transport options
            - Local tips and safety considerations

            ---

            🧠 Plan Smartly:
            - Check what's currently open, practical, and enjoyable
            - Recommend both popular and hidden gems
            - Suggest indoor alternatives if weather is unsuitable
            - Group locations by proximity for efficient travel
            - Highlight travel ease, accessibility, and walking distance where relevant
            - Use web search or tools where necessary to fetch **live data**, refine search queries to help currate itinerary based onr recent data

            ---

            📌 Tailor all recommendations strictly based on the user's travel type (e.g., solo, couple, family, business, group) and budget preferences. It directly affects the style, comfort level, privacy needs, activity preferences, pace of itinerary. All suggestions should reflect what suits the given travel type best.
            
            ---
            **USER QUERY**: {query}
            **TRAVEL TYPE** {travel_type}
            **CURRENCY** {currency} (the currency to which user wants final total expenses to be converted in)
            ---
            
            🔧 Available Tools: {tools}

            🚨 Important: Regardless of how knowledgeable or advanced you are, you may lack access to recent, localized, or dynamically changing data such as updated entry fees, seasonal activity availability, opening hours, or newly popular spots.
            To ensure the accuracy, freshness, and relevance of recommendations, web search is essential and must always be used when curating recommendations. Carry out web search everytime to get more broad variety of data.

            ✍️ **RESPONSE FORMAT**

            ---

            ## 🌤️ CURRENT WEATHER CONDITIONS
            - Weather summary and temperature
            - Impact on outdoor activities or travel

            ----

            ## 🗓️ Day 1

            ### ✨ **Planned Attractions** (2-3 per day)
            - **Attraction Name** - 
            - **Brief Description** - [Brief description, e.g., “Historic fort with panoramic city views”]
            - **Opening Hours:** 
            - **Visit Duration:**  
            - **Entry Fees:** 
            - **Weather Suitability:** 

            ### 🎯 **Planned Activities** (1-2 per day)
            - **[Activity 1]** - [Short description]
            - **Type:** Outdoor/Indoor  
            - **Equipment:**  
            - **Cost:** 
            - **Weather-based Recommendation:** 

            ---

            ## 🗓️ Day 2

            ...repeat similar formatting...

            ---

            ## 🚗 **Transportation & Local Travel**
            - **Recommended Modes:**  
            - **Estimated Costs:** 
            - **Travel Time Between Spots:**
            - **Accessibility Tips:**
            - ⚠️ *If transport info is incomplete, consider looking up online options for local transports, passes or rental services.*

            ---

            ## 💰 **Total Approximate Expense**

            Summarize the estimated expenses for the entire trip based on all the planned components. Break the costs down into the following categories:

            - **Attractions & Activities:** Add the entry fees, activity charges, and any rentals involved across all days.
            - **Transport:** Include costs of public transport, ride-sharing services, bike rentals, or inter-attraction travel.
            - **Buffer (10%):** Add a 10% contingency buffer on top of the total to account for unplanned or miscellaneous expenses.

            ---

            ### ✅ **Total Estimated Cost in [Base Currency Full Name]**
            - Show the full calculated total in the **base currency** used while overall planning above (e.g., *South Korean Won*, *Indian Rupee*).
            - Make sure to label the amount with the **full name of the base currency** for clarity.

            ### 🌍 **Converted Total in [User-Requested Currency Full Name]**
            - Convert the total cost into the **user-requested currency**, as provided in the input variable (`currency`).
            - Show the full name of the target currency (e.g., *US Dollar*, *Euro*, *Indian Rupee*) instead of just symbols or codes.
            - Use a realistic and current conversion rate or approximate exchange value.
            - Clearly mention both values and label them properly for easy understanding.

            ---

            ## 🎒 **Additional Tips**
            - Suggest the best times to visit crowded places
            - Include food/snack options nearby if possible
            - Mention anything the user should carry (ID, water, sunscreen, etc.) 

            ✅ Be thoughtful and detail oriented, offering a seamless and enriching travel experience.
        """,

        input_variables=["query", "travel_type", "currency"],
        partial_variables={"tools": self.required_tools}
        )

        chain = prompt | self.llm_with_tools

        response = chain.invoke({
            "query" : state["messages"],
            "travel_type" : state["travel_type"],
            "currency" : state["currency"]
        })

        return {'messages': [response], 'attraction_result' : response.content}