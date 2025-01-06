from phi.agent import Agent
from phi.model.groq import Groq
import phi.api
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import os
from dotenv import load_dotenv
load_dotenv()
phi.api=os.getenv("PHI_API_KEY")



## web search agent
from phi.agent import Agent
from phi.tools.wikipedia import WikipediaTools

web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Alway include sources"],
    show_tools_calls=True,
    markdown=True,

)

wiki_search_agent = Agent(name = "Wikipedia Search Agent",
    role = "Search Wikipedia for the information",
    model = Groq(id="llama-3.1-70b-versatile"),
    tools=[WikipediaTools()], 
    show_tool_calls=True,
    markdown=True)


finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)



multi_ai_agent=Agent(
    team=[web_search_agent,wiki_search_agent,finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

#multi_ai_agent.print_response("Search M&M.NS stock price , analyst recommendations, stock fundamentals, and company news", stream=True)
z= input("Enter the comapnay name: ")
multi_ai_agent.print_response(f"Search {z} stock price , analyst recommendations, stock fundamentals include opening price , yestarday's opening price, yesterday's closing price, and company news", stream=True)
