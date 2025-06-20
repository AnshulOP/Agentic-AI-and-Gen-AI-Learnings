# ✈️ Assignment: AI Travel Agent & Expense Planner

**Purpose:**  
Design and implement an AI-based Travel Agent that helps users plan a trip to any city in the world using **real-time data**. The system should generate a complete travel itinerary, calculate expenses, and handle currency conversion dynamically.

---

## ✅ Functional Requirements

The AI system should perform the following tasks:

1. **Real-Time Weather Information**
   - Fetch current weather
   - Fetch forecast for the trip duration

2. **Attractions & Activities Discovery**
   - Search top attractions
   - Discover popular restaurants
   - Recommend local activities
   - Explore transportation options

3. **Hotel Cost Estimation**
   - Search for hotels in the selected city
   - Estimate hotel cost: `Cost per day × Total days`
   - Allow input of user’s budget range

4. **Currency Conversion**
   - Get real-time exchange rates
   - Convert total cost to user’s native currency

5. **Expense Calculation**
   - Add and multiply costs for:
     - Hotel
     - Food
     - Transportation
     - Activities
   - Calculate:
     - Daily budget
     - Total trip cost

6. **Itinerary Generation**
   - Create day-wise travel plans
   - Generate a complete itinerary based on preferences and availability

7. **Trip Summary**
   - Summarize selected attractions, costs, weather, and itinerary

8. **Final Output**
   - Return the complete travel plan in a user-friendly format

---

## 🧠 System Flow (Functional Steps)

```text
user_input
   |
   ├── Search Attractions & Activities
   │     ├── Search Attractions
   │     ├── Search Restaurants
   │     ├── Search Activities
   │     └── Search Transportation
   |
   ├── Search Weather Forecasting
   │     ├── Get Current Weather
   │     └── Get Weather Forecast
   |
   ├── Hotel Cost Estimation
   │     ├── Search Hotels
   │     ├── Estimate Hotel Cost
   │     └── Input Budget Range
   |
   ├── Calculate Total Cost
   │     ├── Add Individual Costs
   │     ├── Multiply Unit Costs
   │     ├── Compute Total Trip Cost
   │     └── Calculate Daily Budget
   |
   ├── Currency Conversion
   │     ├── Get Exchange Rate
   │     └── Convert Currency
   |
   ├── Itinerary Generation
   │     ├── Get Day Plan
   │     └── Create Full Itinerary
   |
   ├── Create Trip Summary
   │
   └── Return Final Travel Plan
