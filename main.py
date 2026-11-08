import os
import io
import csv
import json
import re
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
from bs4 import BeautifulSoup
from ddgs import DDGS

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.flow.flow import Flow, listen, or_, router, start
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field


# Gemini configuration: set GEMINI_API_KEY in the hosting service's environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it as an environment variable in Render."
    )

llm = LLM(
    model="gemini/gemini-3.5-flash-lite",
    api_key=GEMINI_API_KEY,
    temperature=0.3
)


# -------------------- AGENT 1: BLOG POST CREW --------------------

from crewai.tools import tool
from ddgs import DDGS
from pathlib import Path

@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web and return relevant titles, links and summaries."""
    try:
        results = DDGS().text(query, max_results=5)
        output = []

        for r in results:
            output.append(
                f"Title: {r.get('title', '')}\n"
                f"Link: {r.get('href', '')}\n"
                f"Summary: {r.get('body', '')}"
            )

        return "\n\n".join(output) if output else "No results found"

    except Exception as e:
        return f"Search failed: {e}"


@tool("File Writer")
def write_file(filename: str, content: str) -> str:
    """Save the final blog post to a markdown file."""
    try:
        Path(filename).write_text(content, encoding="utf-8")
        return f"File saved successfully: {filename}"
    except Exception as e:
        return f"File writing failed: {e}"


researcher = Agent(
    role="Blog Researcher",
    goal="Research the given topic thoroughly and collect relevant facts, recent developments and examples with source links",
    backstory="You are a researcher who gathers reliable information before a blog post is written",
    llm=llm,
    tools=[duckduckgo_search],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

writer = Agent(
    role="Blog Writer",
    goal="Write an engaging and informative blog post of approximately 600 words",
    backstory="You are an experienced content writer who transforms research into clear, useful and interesting articles.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

editor = Agent(
    role="Blog Editor",
    goal="Edit the draft for grammar, clarity, readability and factual consistency and save the final markdown file",
    backstory="You are a professional editor who improves content quality without changing the meaning",
    llm=llm,
    tools=[write_file],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

research_task = Task(
    description="""Research the topic {topic} using DuckDuckGo Search.
    Search specifically for {topic}.
    Collect important facts, recent developments, examples and relevant source links.
    Do not include information unrelated to the given topic.
    Do not invent statistics or sources.
    Present the research in a structured format.""",
    expected_output="A structured research report containing key findings, examples and source links.",
    agent=researcher
)

writing_task = Task(
    description="""Write a blog post on {topic} with approximately 600 words.
    Include an attractive title, introduction, clear headings, examples and a conclusion.
    Use simple, professional and reader-friendly language.""",
    expected_output="A complete blog post in Markdown, approximately 600 words, based on the research report.",
    agent=writer,
    context=[research_task]
)

editing_task = Task(
    description="""Edit the blog post for grammar, clarity, readability and factual accuracy.
    Preserve source links, remove unnecessary wording, and save the final article as blog_post.md using the File Writer tool.""",
    expected_output="The final edited blog post and save confirmation.",
    agent=editor,
    context=[research_task, writing_task]
)

blog_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=10
)

print("Blog Post Crew created successfully!")

# -------------------- AGENT 2: COMPETITOR ANALYSIS CREW --------------------

from crewai.tools import tool
from ddgs import DDGS
from pathlib import Path
import requests
from bs4 import BeautifulSoup


@tool("Scrape Competitor Website")
def competitor_scrape_website(url: str) -> str:
    """Extract readable text from a competitor website URL."""
    try:
        if not url.startswith(("http://", "https://")):
            return "Invalid URL. Use http:// or https://."

        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(
            ["script", "style", "nav", "footer", "noscript"]
        ):
            element.decompose()

        return soup.get_text(separator=" ", strip=True)[:12000]

    except Exception as e:
        return f"Website scraping failed: {e}"


@tool("Competitor DuckDuckGo Search")
def competitor_duckduckgo_search(query: str) -> str:
    """Search for competitor news, updates and public information."""
    try:
        results = DDGS().text(query, max_results=5)
        output = []

        for r in results:
            output.append(
                f"Title: {r.get('title', '')}\n"
                f"Link: {r.get('href', '')}\n"
                f"Summary: {r.get('body', '')}"
            )

        return "\n\n".join(output) if output else "No results found."

    except Exception as e:
        return f"Search failed: {e}"


@tool("Competitor File Writer")
def competitor_write_file(filename: str, content: str) -> str:
    """Save the final competitor analysis report as a Markdown file."""
    try:
        Path(filename).write_text(content, encoding="utf-8")
        return f"File saved successfully: {filename}"
    except Exception as e:
        return f"File writing failed: {e}"


site_scraper = Agent(
    role="Site Scraper",
    goal="Extract relevant product, service and feature information from competitor websites.",
    backstory="You are a website research specialist who gathers useful public information.",
    llm=llm,
    tools=[competitor_scrape_website],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

pricing_analyst = Agent(
    role="Pricing Analyst",
    goal="Compare competitor pricing, plans, features and value propositions.",
    backstory="You analyze pricing models and product offerings to identify competitive differences.",
    llm=llm,
    tools=[competitor_scrape_website],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

news_analyst = Agent(
    role="News Analyst",
    goal="Find relevant competitor news, launches, updates and market developments.",
    backstory="You research public news and distinguish sourced facts from assumptions.",
    llm=llm,
    tools=[competitor_duckduckgo_search],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

strategist = Agent(
    role="Competitor Strategist",
    goal="Combine competitor research into a comparative analysis and SWOT report.",
    backstory="You are a business strategist who turns competitive research into actionable insights.",
    llm=llm,
    tools=[competitor_write_file],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

competitor_research_task = Task(
    description="""
    Analyze the following competitor websites: {competitors}

    Visit each website using the Scrape Competitor Website tool.
    Identify the company's products, services, target customers,
    features and unique selling points.

    Clearly identify which competitor each finding belongs to.
    Include source URLs and do not invent missing information.
    """,
    expected_output="A competitor-by-competitor website research report with source URLs.",
    agent=site_scraper
)

competitor_pricing_task = Task(
    description="""
    Analyze the competitors: {competitors}

    Use the Scrape Competitor Website tool to investigate their pricing,
    subscription plans, product features and value propositions.

    Compare the available information in a table.
    Mark unavailable pricing as 'Not found' rather than guessing.
    Include source URLs for the findings.
    """,
    expected_output="A comparative pricing and features table with source URLs.",
    agent=pricing_analyst
)

competitor_news_task = Task(
    description="""
    Research recent news and public developments about these
    competitors: {competitors}

    Use DuckDuckGo Search to find product launches, partnerships,
    company announcements and market developments.

    Include dates and source links when available.
    Do not invent news or treat undated information as recent.
    """,
    expected_output="A competitor news summary with dates and source links.",
    agent=news_analyst
)

competitor_strategy_task = Task(
    description="""
    Using the website research, pricing analysis and news findings,
    prepare a comprehensive competitor analysis for: {competitors}

    Include:
    1. Executive summary
    2. Competitor overview
    3. Products and features comparison
    4. Pricing comparison
    5. Recent news and developments
    6. SWOT analysis for each competitor, where evidence permits
    7. Key competitive advantages and gaps
    8. Actionable recommendations
    9. Source links

    Distinguish verified facts from strategic interpretations.
    Do not invent prices, facts or sources.

    Save the complete final report as report.md using the File Writer
    tool, then return the report and confirm the saved filename.
    """,
    expected_output="A complete competitor analysis report in Markdown saved as report.md.",
    agent=strategist,
    context=[
        competitor_research_task,
        competitor_pricing_task,
        competitor_news_task
    ]
)

competitor_crew = Crew(
    agents=[
        site_scraper,
        pricing_analyst,
        news_analyst,
        strategist
    ],
    tasks=[
        competitor_research_task,
        competitor_pricing_task,
        competitor_news_task,
        competitor_strategy_task
    ],
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True,
    max_rpm=10
)

print("Competitor Analysis Crew created successfully!")

# -------------------- AGENT 3: MEETING NOTES CREW --------------------

from crewai.tools import tool
from pathlib import Path

@tool("Meeting File Read")
def meeting_read_file(filename: str) -> str:
    """Read meeting transcript from a text file."""
    try:
        return Path(filename).read_text(encoding="utf-8")
    except Exception as e:
        return f"File reading failed: {e}"


@tool("Meeting File Writer")
def meeting_write_file(filename: str, content: str) -> str:
    """Save the final meeting notes to a Markdown file."""
    try:
        Path(filename).write_text(content, encoding="utf-8")
        return f"File saved successfully: {filename}"
    except Exception as e:
        return f"File writing failed: {e}"


summarizer = Agent(
    role="Meeting Summarizer",
    goal="Summarize the meeting transcript in five concise lines.",
    backstory="You create clear and accurate meeting summaries.",
    llm=llm,
    tools=[meeting_read_file],
    verbose=True,
    allow_delegation=False
)

action_finder = Agent(
    role="Action Item Finder",
    goal="Identify action items, responsible owners and due dates.",
    backstory="You extract tasks and deadlines from meeting discussions.",
    llm=llm,
    tools=[meeting_read_file],
    verbose=True,
    allow_delegation=False
)

decision_logger = Agent(
    role="Decision Logger",
    goal="Identify important decisions made during the meeting.",
    backstory="You record meeting decisions accurately without inventing details.",
    llm=llm,
    tools=[meeting_read_file],
    verbose=True,
    allow_delegation=False
)

formatter = Agent(
    role="Meeting Notes Formatter",
    goal="Combine the summary, action items and decisions into notes.md.",
    backstory="You organize meeting information into professional Markdown notes.",
    llm=llm,
    tools=[meeting_write_file],
    verbose=True,
    allow_delegation=False
)

meeting_summary_task = Task(
    description="""
    Read the transcript from {filename} using the Meeting File Read tool.
    Summarize the meeting in five concise lines.
    """,
    expected_output="A five-line meeting summary.",
    agent=summarizer
)

meeting_action_task = Task(
    description="""
    Read {filename} using the Meeting File Read tool.
    Identify all action items, their owners and due dates.
    Use 'Not specified' when information is missing.
    """,
    expected_output="A list of action items, owners and due dates.",
    agent=action_finder,
    context=[meeting_summary_task]
)

meeting_decision_task = Task(
    description="""
    Read {filename} using the Meeting File Read tool.
    Record the important decisions made during the meeting.
    Do not invent decisions.
    """,
    expected_output="A clear list of meeting decisions.",
    agent=decision_logger,
    context=[meeting_summary_task]
)

meeting_format_task = Task(
    description="""
    Combine the meeting summary, action items and decisions.
    Format them with Markdown headings and save the complete
    notes as notes.md using the Meeting File Writer tool.
    """,
    expected_output="Formatted meeting notes saved as notes.md.",
    agent=formatter,
    context=[
        meeting_summary_task,
        meeting_action_task,
        meeting_decision_task
    ]
)

meeting_crew = Crew(
    agents=[
        summarizer,
        action_finder,
        decision_logger,
        formatter
    ],
    tasks=[
        meeting_summary_task,
        meeting_action_task,
        meeting_decision_task,
        meeting_format_task
    ],
    process=Process.sequential,
    verbose=True,
    max_rpm=10
)

print("Meeting Notes Crew created successfully!")

# -------------------- AGENT 4: DATA REPORT CREW --------------------

from crewai.tools import tool
from pathlib import Path
import pandas as pd


@tool("Data CSV File Reader")
def data_read_csv_file(filename: str) -> str:
    """Read a CSV file and return its statistical analysis."""
    try:
        df = pd.read_csv(filename)

        result = []

        result.append("DATASET OVERVIEW")
        result.append(f"Total rows: {len(df)}")
        result.append(f"Total columns: {len(df.columns)}")
        result.append(f"Column names: {list(df.columns)}")

        result.append("\nSTATISTICAL SUMMARY")
        result.append(df.describe(include="all").to_string())

        result.append("\nMISSING VALUES")
        result.append(df.isnull().sum().to_string())

        result.append("\nFIRST FIVE ROWS")
        result.append(df.head().to_string())

        return "\n".join(result)

    except Exception as e:
        return f"CSV reading failed: {e}"


@tool("Data Report File Writer")
def data_write_report(filename: str, content: str) -> str:
    """Save the final data analysis report as a Markdown file."""
    try:
        Path(filename).write_text(content, encoding="utf-8")
        return f"File saved successfully: {filename}"

    except Exception as e:
        return f"File writing failed: {e}"


data_analyst = Agent(
    role="Data Analyst",
    goal="Analyze the CSV dataset and identify important statistical information.",
    backstory="You are an experienced data analyst who uses pandas to examine datasets and discover numerical patterns.",
    llm=llm,
    tools=[data_read_csv_file],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

insight_writer = Agent(
    role="Insight Writer",
    goal="Identify the top 3 important findings from the analyzed dataset.",
    backstory="You are a business insight specialist who converts statistical information into meaningful findings.",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5
)

data_reviewer = Agent(
    role="Data Report Reviewer",
    goal="Review the analysis and insights for accuracy and prepare the final report.",
    backstory="You are a professional reviewer who checks numerical accuracy, clarity and consistency before publishing reports.",
    llm=llm,
    tools=[data_write_report],
    verbose=True,
    allow_delegation=False,
    max_iter=5
)


data_analysis_task = Task(
    description="""
    Read and analyze the CSV file from {filename}
    using the Data CSV File Reader tool.

    Identify:
    1. Total number of rows and columns.
    2. Column names.
    3. Statistical summary.
    4. Missing values.
    5. Important numerical patterns.

    Do not invent any numerical values.
    """,
    expected_output="A detailed statistical analysis of the CSV dataset.",
    agent=data_analyst
)

data_insight_task = Task(
    description="""
    Study the analysis provided by the Data Analyst.

    Identify the TOP 3 important findings.

    For each finding explain:
    1. What was observed.
    2. The supporting numerical value.
    3. Why the finding is important.

    Use only the information provided by the Data Analyst.
    Do not invent statistics.
    """,
    expected_output="Three important findings with numerical evidence.",
    agent=insight_writer,
    context=[data_analysis_task]
)

data_review_task = Task(
    description="""
    Review the statistical analysis and top 3 findings.

    Check the numerical values and remove unsupported claims.

    Prepare a professional Markdown report containing:

    1. Dataset Overview
    2. Statistical Summary
    3. Missing Values
    4. Top 3 Findings
    5. Conclusion

    Save the final report as report.md using the Data Report File Writer tool.

    Return the final report and confirm the saved filename.
    """,
    expected_output="A reviewed and accurate data analysis report saved as report.md.",
    agent=data_reviewer,
    context=[data_analysis_task, data_insight_task]
)


data_report_crew = Crew(
    agents=[data_analyst, insight_writer, data_reviewer],
    tasks=[data_analysis_task, data_insight_task, data_review_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=10
)

print("Data Report Crew created successfully!")

# -------------------- AGENT 5: CUSTOMER SUPPORT FLOW --------------------

# AGENT 5: CUSTOMER SUPPORT FLOW

import re
from pydantic import BaseModel, Field
from crewai.flow.flow import Flow, listen, or_, router, start


# 1. Define Flow State
class SupportState(BaseModel):
    ticket: str = ""
    category: str = ""
    priority: str = ""
    handling_notes: str = ""
    resolution: str = ""


# 2. Define Customer Support Flow
class CustomerSupportFlow(Flow[SupportState]):

    # Receive customer support ticket
    @start()
    def receive_ticket(self):

        if not self.state.ticket.strip():
            raise ValueError("Support ticket cannot be empty.")

        print("\nTicket received successfully.")
        print("Ticket:", self.state.ticket)

        return self.state.ticket


    # Categorize ticket using Gemini
    @listen(receive_ticket)
    def categorize_ticket(self, ticket):

        prompt = f"""
        You are a customer support ticket classifier.

        Classify the following ticket into exactly
        ONE of these categories:

        1. billing - Payment, refund, invoice, transaction issues.
        2. account_access - Login, password, account locked,
           authentication issues.
        3. other - Technical issues, bugs, features,
           and all other requests.

        Return ONLY one category label:
        billing
        account_access
        other

        Customer Ticket:
        {ticket}
        """

        response = str(llm.call(prompt)).strip().lower()

        match = re.search(
            r"\b(billing|account_access|other)\b",
            response
        )

        if not match:
            raise ValueError("Could not identify ticket category.")

        self.state.category = match.group(1)

        print("\nTicket categorized successfully.")
        print("Category:", self.state.category)

        return self.state.category


    # 3. Router: Route ticket based on category
    @router(categorize_ticket)
    def route_by_category(self, category):

        print("\n========== ROUTING TICKET ==========")

        if category == "billing":
            print("Routing to Billing Support.")
            return "billing"

        elif category == "account_access":
            print("Routing to Account Access Support.")
            return "account_access"

        else:
            print("Routing to General Technical Support.")
            return "other"


    # 4. Handle Billing Tickets
    @listen("billing")
    def handle_billing(self):

        self.state.priority = "High"

        self.state.handling_notes = """
        Handle this as a billing-related issue.
        Check payment status, transaction details,
        invoices, and refund-related concerns.
        Never claim a refund or payment has been processed
        without actual confirmation.
        """

        print("\nBilling Support Handler Activated.")
        print("Priority: High")

        return "Billing handled"


    # 5. Handle Account Access Tickets
    @listen("account_access")
    def handle_account_access(self):

        self.state.priority = "High"

        self.state.handling_notes = """
        Handle this as an account access issue.
        Provide password reset instructions,
        login troubleshooting, and account recovery guidance.
        Never request passwords or OTPs from the customer.
        """

        print("\nAccount Access Handler Activated.")
        print("Priority: High")

        return "Account access handled"


    # 6. Handle Other Tickets
    @listen("other")
    def handle_other(self):

        self.state.priority = "Normal"

        self.state.handling_notes = """
        Handle this as a general technical or feature-related issue.
        Understand the problem, suggest troubleshooting steps,
        and provide relevant technical guidance.
        Escalate complex unresolved problems when necessary.
        """

        print("\nGeneral Support Handler Activated.")
        print("Priority: Normal")

        return "Other issue handled"


    # 7. Resolve ticket after any handler completes
    @listen(or_(
        handle_billing,
        handle_account_access,
        handle_other
    ))
    def resolve_ticket(self, outcome):

        prompt = f"""
        You are a professional customer support assistant.

        Generate a helpful and professional response
        to resolve the customer's support ticket.

        Customer Ticket:
        {self.state.ticket}

        Category:
        {self.state.category}

        Priority:
        {self.state.priority}

        Handling Instructions:
        {self.state.handling_notes}

        Requirements:
        1. Address the customer's actual problem.
        2. Provide clear and simple steps.
        3. Be polite and professional.
        4. Do not invent transaction or account details.
        5. Do not claim that an action was completed
           if it has not actually been performed.
        6. If the issue cannot be resolved, suggest
           contacting the appropriate support team.

        Generate the final customer support response.
        """

        self.state.resolution = str(
            llm.call(prompt)
        ).strip()

        print("\n========== TICKET RESOLUTION ==========")
        print(self.state.resolution)

        return self.state.resolution


# 8. Create Flow Instance
support_flow = CustomerSupportFlow()

print("Customer Support Flow (Agent 5) created successfully!")

# -------------------- AGENT 6: RESUME SCREENING FLOW --------------------

# AGENT 6: RESUME SCREENING FLOW

import json
import re
from pydantic import BaseModel, Field
from crewai.flow.flow import Flow, listen, or_, router, start


# 1. Define Flow State
class ResumeState(BaseModel):
    resume_text: str = ""
    filename: str = "Candidate Resume"
    score: int = 0
    summary: str = ""
    rejection_email: str = ""
    decision: str = ""
    results: list = Field(default_factory=list)


# 2. Define Resume Screening Flow
class ResumeScreeningFlow(Flow[ResumeState]):

    # Receive resume
    @start()
    def load_resume(self):

        if not self.state.resume_text.strip():
            raise ValueError("Resume text cannot be empty.")

        print("\nResume received successfully.")

        return self.state.resume_text


    # 3. Score Resume using Gemini
    @listen(load_resume)
    def score_resume(self, resume_text):

        prompt = f"""
        You are assisting a recruiter screening candidates
        for a Software Developer role.

        Evaluate the resume based on relevant programming
        skills, SQL, DSA, education, and job-related experience.

        Give a score from 0 to 10 based on role-related evidence.
        Do not infer protected personal characteristics.
        Do not penalize candidates for personal characteristics
        unrelated to job performance.

        Return ONLY valid JSON in this format:

        {{
            "score": 0,
            "summary": "Brief strengths and skill gaps"
        }}

        Resume:
        {resume_text[:12000]}
        """

        try:
            response = llm.call(prompt)

            match = re.search(
                r"\{.*\}", str(response), re.DOTALL
            )

            if not match:
                raise ValueError("No JSON returned")

            data = json.loads(match.group())

            score = int(data["score"])

            if not 0 <= score <= 10:
                raise ValueError("Score outside 0-10")

            self.state.score = score
            self.state.summary = str(data.get("summary", ""))

            self.state.results = [{
                "filename": self.state.filename,
                "score": score,
                "summary": self.state.summary
            }]

        except Exception as error:
            raise ValueError(f"Resume scoring failed: {error}")

        print("\nResume scoring completed.")
        print("Score:", self.state.score)

        return self.state.score


    # 4. Router: Shortlist or Reject
    @router(score_resume)
    def check_score(self, score):

        if score >= 7:
            print("\nCandidate Shortlisted.")
            return "shortlist"

        else:
            print("\nCandidate Rejected.")
            return "reject"


    # 5. Create Shortlist Summary
    @listen("shortlist")
    def summarize_resume(self):

        self.state.decision = "Shortlisted"

        print("\nCandidate Shortlisted.")
        print("Score:", self.state.score, "/10")
        print("Summary:", self.state.summary)

        return "shortlist"


    # 6. Draft Rejection Email
    @listen("reject")
    def draft_rejection(self):

        prompt = f"""
        Write a short, polite, professional rejection email
        for a Software Developer applicant.

        Do not invent the candidate's name or contact details.
        Refer to the recipient as Candidate.
        Do not include unsupported reasons for rejection.

        Screening score: {self.state.score}/10

        Resume Summary:
        {self.state.summary}
        """

        self.state.rejection_email = str(
            llm.call(prompt)
        ).strip()

        self.state.decision = "Rejected"

        print("\nCandidate Rejected.")
        print("Draft email:")
        print(self.state.rejection_email)

        return "reject"


    # 7. Prepare Final Screening Results
    @listen(or_(summarize_resume, draft_rejection))
    def save_results(self, outcome):

        result = f"""
        RESUME SCREENING REPORT

        Resume: {self.state.filename}
        Score: {self.state.score}/10
        Decision: {self.state.decision}

        Summary:
        {self.state.summary}

        Rejection Email:
        {self.state.rejection_email or "Not required."}
        """

        print("\n========== FINAL SCREENING RESULTS ==========")
        print(result)

        return result


# 8. Create Flow Instance
resume_flow = ResumeScreeningFlow()

print("Resume Screening Flow (Agent 6) created successfully!")

# -------------------- AGENT 7: BUG TRIAGE FLOW --------------------

# AGENT 7: BUG TRIAGE FLOW

import csv
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen, router, or_


class BugTriageState(BaseModel):
    issue_title: str = ""
    issue_body: str = ""
    issue_type: str = ""
    result: str = ""
    label: str = ""
    results: list = Field(default_factory=list)


# Crew 1: Bug Analysis Crew
bug_analyst = Agent(
    role="Bug Triage Analyst",
    goal="Analyze bug severity and recommend debugging steps",
    backstory="An experienced software engineer who investigates bugs.",
    llm=llm,
    verbose=False
)

bug_task = Task(
    description="""
    Analyze this issue.

    Title: {title}
    Description: {body}

    Identify the severity as Critical, High, Medium, or Low.
    Explain the likely cause if possible and provide practical
    debugging steps. Do not invent facts.
    """,
    expected_output="Bug severity, analysis, and debugging steps.",
    agent=bug_analyst
)

bug_crew = Crew(
    agents=[bug_analyst],
    tasks=[bug_task],
    process=Process.sequential,
    verbose=False
)


# Crew 2: Feature Request Crew
feature_analyst = Agent(
    role="Feature Analyst",
    goal="Convert feature requests into clear user stories",
    backstory="An experienced product analyst who defines requirements.",
    llm=llm,
    verbose=False
)

feature_task = Task(
    description="""
    Convert this feature request into a user story.

    Title: {title}
    Description: {body}

    Include:
    1. User story in the format:
       As a user, I want ..., so that ...
    2. Acceptance criteria.
    """,
    expected_output="A user story with acceptance criteria.",
    agent=feature_analyst
)

feature_crew = Crew(
    agents=[feature_analyst],
    tasks=[feature_task],
    process=Process.sequential,
    verbose=False
)


# Crew 3: Documentation Answer Crew
answer_analyst = Agent(
    role="Documentation Support Analyst",
    goal="Answer questions using the provided documentation",
    backstory="A technical support specialist who explains documentation.",
    llm=llm,
    verbose=False
)

answer_task = Task(
    description="""
    Answer the user's question using ONLY the documentation below.

    Issue title: {title}
    Question: {body}

    Documentation:
    {docs}

    If the documentation does not contain the answer, clearly
    say that the available documentation is insufficient.
    Do not invent documentation or citations.
    """,
    expected_output="A concise answer grounded in the documentation.",
    agent=answer_analyst
)

answer_crew = Crew(
    agents=[answer_analyst],
    tasks=[answer_task],
    process=Process.sequential,
    verbose=False
)


# Bug Triage Flow
class BugTriageFlow(Flow[BugTriageState]):

    @start()
    def read_issue(self):
        if not self.state.issue_body.strip():
            raise ValueError("Issue description cannot be empty.")

        if not self.state.issue_title.strip():
            self.state.issue_title = "Software Issue"

        return "Issue received"

    @listen(read_issue)
    def classify_issue(self, _):
        prompt = f"""
        Classify this software issue into exactly ONE category:
        bug, feature, or question.

        Title: {self.state.issue_title}
        Description: {self.state.issue_body}

        bug = existing functionality is broken.
        feature = request for new or improved functionality.
        question = request for information or help.

        Return only one category word.
        """

        response = str(llm.call(prompt)).strip().lower()
        category = response.split()[0].strip(".,:;`'\"") if response else ""

        if category not in ["bug", "feature", "question"]:
            category = "question"

        self.state.issue_type = category
        print("Classified as:", category)
        return category

    @router(classify_issue)
    def route_type(self, category):
        return category

    # Bug branch
    @listen("bug")
    def run_bug_crew(self):
        result = bug_crew.kickoff(inputs={
            "title": self.state.issue_title,
            "body": self.state.issue_body
        })

        self.state.label = "bug"
        self.state.result = str(result)
        return self.state.result

    # Feature branch
    @listen("feature")
    def run_feature_crew(self):
        result = feature_crew.kickoff(inputs={
            "title": self.state.issue_title,
            "body": self.state.issue_body
        })

        self.state.label = "feature"
        self.state.result = str(result)
        return self.state.result

    # Question branch
    @listen("question")
    def run_answer_crew(self):
        docs = """
        Login troubleshooting:
        1. Verify that the username and password are correct.
        2. Check whether the authentication service is available.
        3. Review application logs for authentication errors.
        """

        result = answer_crew.kickoff(inputs={
            "title": self.state.issue_title,
            "body": self.state.issue_body,
            "docs": docs
        })

        self.state.label = "question"
        self.state.result = str(result)
        return self.state.result

    # Merge all three branches
    @listen(or_(run_bug_crew, run_feature_crew, run_answer_crew))
    def post_reply(self, _):
        result = {
            "title": self.state.issue_title,
            "body": self.state.issue_body,
            "label": self.state.label,
            "response": self.state.result
        }

        self.state.results.append(result)

        with open(
            "bug_triage_results.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["title", "body", "label", "response"]
            )
            writer.writeheader()
            writer.writerows(self.state.results)

        print("Issue label:", self.state.label)
        print("Crew response:\n", self.state.result)

        return self.state.result


triage_flow = BugTriageFlow()

print("Bug Triage Flow (Agent 7) created successfully!")

# -------------------- AGENT 8: CONTENT REPURPOSING FLOW --------------------

# AGENT 8: CONTENT REPURPOSING FLOW

from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor


# Flow state
class ContentState(BaseModel):
    article: str = ""
    key_points: str = ""
    linkedin_post: str = ""
    tweet_thread: str = ""
    newsletter: str = ""
    output_files: list = Field(default_factory=list)


# Crew 1: LinkedIn Crew
linkedin_agent = Agent(
    role="LinkedIn Content Writer",
    goal="Create an engaging LinkedIn post of about 150 words",
    backstory="An expert at writing professional social media content.",
    llm=llm,
    verbose=False
)

linkedin_task = Task(
    description="""
    Using these article key points:
    {points}

    Write a LinkedIn post of approximately 150 words.
    Start with a strong hook, explain the main insights,
    and finish with a meaningful takeaway.
    Do not invent facts.
    """,
    expected_output="One polished LinkedIn post of approximately 150 words.",
    agent=linkedin_agent
)

linkedin_crew = Crew(
    agents=[linkedin_agent],
    tasks=[linkedin_task],
    process=Process.sequential,
    verbose=False
)


# Crew 2: Twitter Crew
twitter_agent = Agent(
    role="Twitter Thread Writer",
    goal="Turn article insights into a clear six-tweet thread",
    backstory="An expert at concise and engaging social content.",
    llm=llm,
    verbose=False
)

twitter_task = Task(
    description="""
    Using these article key points:
    {points}

    Create exactly six numbered tweets.
    Each tweet must be concise and understandable.
    Make the thread flow naturally from introduction
    to insights and a final takeaway.
    Do not invent facts.
    """,
    expected_output="A numbered thread containing exactly six tweets.",
    agent=twitter_agent
)

twitter_crew = Crew(
    agents=[twitter_agent],
    tasks=[twitter_task],
    process=Process.sequential,
    verbose=False
)


# Crew 3: Newsletter Crew
newsletter_agent = Agent(
    role="Newsletter Writer",
    goal="Create a newsletter introduction and three key bullets",
    backstory="An expert at turning long articles into useful newsletters.",
    llm=llm,
    verbose=False
)

newsletter_task = Task(
    description="""
    Using these article key points:
    {points}

    Write:
    1. A short newsletter introduction.
    2. Exactly three useful bullet points.
    3. A brief closing takeaway.

    Stay faithful to the source material.
    """,
    expected_output="Newsletter introduction, three bullets, and closing.",
    agent=newsletter_agent
)

newsletter_crew = Crew(
    agents=[newsletter_agent],
    tasks=[newsletter_task],
    process=Process.sequential,
    verbose=False
)


# Content Repurposing Flow
class ContentRepurposingFlow(Flow[ContentState]):

    @start()
    def load_article(self):
        if not self.state.article.strip():
            raise ValueError("Article text cannot be empty.")

        self.state.article = self.state.article[:25000]

        print("Article loaded successfully.")
        return "Article loaded"

    @listen(load_article)
    def extract_points(self, _):
        prompt = f"""
        Extract the main ideas from this article.

        Return:
        - Main topic
        - Central argument
        - Important facts and insights
        - Practical takeaways

        Use only information supported by the article.
        Keep the result concise.

        Article:
        {self.state.article}
        """

        self.state.key_points = str(llm.call(prompt)).strip()
        print("Key points extracted.")

        return self.state.key_points

    # Run all three crews concurrently
    @listen(extract_points)
    def create_content(self, points):

        def create_linkedin():
            return str(
                linkedin_crew.kickoff(inputs={"points": points})
            )

        def create_tweets():
            return str(
                twitter_crew.kickoff(inputs={"points": points})
            )

        def create_newsletter():
            return str(
                newsletter_crew.kickoff(inputs={"points": points})
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            linkedin_future = executor.submit(create_linkedin)
            twitter_future = executor.submit(create_tweets)
            newsletter_future = executor.submit(create_newsletter)

            self.state.linkedin_post = linkedin_future.result()
            self.state.tweet_thread = twitter_future.result()
            self.state.newsletter = newsletter_future.result()

        print("All three content formats generated.")
        return "Content generated"

    @listen(create_content)
    def publish_pack(self):
        outputs = {
            "linkedin_post.txt": self.state.linkedin_post,
            "twitter_thread.txt": self.state.tweet_thread,
            "newsletter.txt": self.state.newsletter
        }

        for filename, content in outputs.items():
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)

        self.state.output_files = list(outputs.keys())

        print("Content repurposing completed.")
        print("Files created:", self.state.output_files)

        return "Three files saved"


content_flow = ContentRepurposingFlow()

print("Content Repurposing Flow (Agent 8) created successfully!")

# -------------------- AGENT 9: LEAD QUALIFICATION FLOW --------------------

# Agent 9 support definitions (missing from the uploaded notebook).
class LeadState(BaseModel):
    lead_input: str = ""
    name: str = ""
    website: str = ""
    website_text: str = ""
    lead_score: int = 0
    category: str = ""
    reason: str = ""
    result: str = ""
    leads: list = Field(default_factory=list)


lead_enrichment_agent = Agent(
    role="Lead Research Analyst",
    goal="Extract relevant business information from a company's website",
    backstory="An analyst who researches companies using public website content.",
    llm=llm,
    verbose=False
)

lead_enrichment_task = Task(
    description="""
    Research this lead using the supplied website text.

    Lead name: {name}
    Website: {website}
    Website text: {website_text}

    Summarize the business, its products or services, target customers,
    and any publicly stated business needs relevant to our offering.

    Do not invent facts. Clearly mention missing information.
    """,
    expected_output="A concise lead research summary grounded in website text.",
    agent=lead_enrichment_agent
)

enrich_crew = Crew(
    agents=[lead_enrichment_agent],
    tasks=[lead_enrichment_task],
    process=Process.sequential,
    verbose=False
)


# Lead Qualification Flow
class LeadQualificationFlow(Flow[LeadState]):

    @start()
    def capture_lead(self):
        user_input = self.state.lead_input.strip()

        if not user_input:
            raise ValueError("Please enter a website name or link.")

        if not user_input.startswith(("http://", "https://")):
            user_input = "https://" + user_input

        parsed_url = urlparse(user_input)

        if not parsed_url.hostname or "." not in parsed_url.hostname:
            raise ValueError("Please enter a valid website name or link.")

        self.state.website = user_input

        domain = parsed_url.hostname.replace("www.", "")
        self.state.name = domain.split(".")[0].capitalize()

        print("Company Name:", self.state.name)
        print("Website:", self.state.website)

        return "Lead captured"

    @listen(capture_lead)
    def enrich_lead(self, _):
        headers = {"User-Agent": "Mozilla/5.0"}

        try:
            response = requests.get(
                self.state.website,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            self.state.website_text = soup.get_text(
                separator=" ",
                strip=True
            )[:10000]

        except requests.RequestException as error:
            print("Website could not be accessed:", error)
            self.state.website_text = ""

        if not self.state.website_text:
            self.state.website_text = (
                "Website content unavailable. Qualification must "
                "be based on limited information."
            )

        research = enrich_crew.kickoff(inputs={
            "name": self.state.name,
            "website": self.state.website,
            "website_text": self.state.website_text
        })

        self.state.result = str(research)

        print("\nLead research completed.")
        return self.state.result

    @listen(enrich_lead)
    def score_lead(self, research):
        prompt = f"""
        Qualify this business lead for a hypothetical B2B software
        product. Do not assume a product fit without evidence.

        Lead: {self.state.name}
        Website: {self.state.website}
        Research: {research}

        Return ONLY valid JSON:
        {{
          "score": 0,
          "category": "hot",
          "reason": "Brief evidence-based explanation"
        }}

        Score from 0 to 100 based on demonstrated relevance,
        apparent need, and evidence of potential fit.

        hot = 70-100
        warm = 40-69
        cold = 0-39

        Use the available evidence only. If information is insufficient,
        choose a conservative score and explain the uncertainty.

        Return category as exactly hot, warm, or cold.
        """

        response = str(llm.call(prompt))
        match = re.search(r"\{.*\}", response, re.DOTALL)

        try:
            if not match:
                raise ValueError("No JSON found")

            data = json.loads(match.group())
            score = int(data["score"])

            if not 0 <= score <= 100:
                raise ValueError("Score must be between 0 and 100")

            self.state.lead_score = score

            self.state.category = (
                "hot" if score >= 70
                else "warm" if score >= 40
                else "cold"
            )

            self.state.reason = str(data.get("reason", ""))

        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.state.lead_score = 0
            self.state.category = "cold"
            self.state.reason = (
                "Unable to validate score; manual review required."
            )

        print("Lead score:", self.state.lead_score)
        print("Category:", self.state.category)
        print("Reason:", self.state.reason)

        return self.state.category

    @router(score_lead)
    def route_lead(self, category):
        return category

    @listen("hot")
    def outreach_hot_lead(self):
        result = outreach_crew.kickoff(inputs={
            "name": self.state.name,
            "website": self.state.website,
            "research": self.state.result
        })

        self.state.result = str(result)
        print("\nOutreach email draft:\n", self.state.result)
        return self.state.result

    @listen("warm")
    def nurture_warm_lead(self):
        result = nurture_crew.kickoff(inputs={
            "name": self.state.name,
            "website": self.state.website,
            "research": self.state.result
        })

        self.state.result = str(result)
        print("\nNurture plan:\n", self.state.result)
        return self.state.result

    @listen("cold")
    def archive_cold_lead(self):
        self.state.result = (
            "Lead archived. No outreach action taken. "
            "Review again if new information becomes available."
        )

        print("\n", self.state.result)
        return self.state.result

    @listen(or_(
        outreach_hot_lead,
        nurture_warm_lead,
        archive_cold_lead
    ))
    def log_lead(self, branch_result):
        self.state.leads.append({
            "name": self.state.name,
            "website": self.state.website,
            "score": self.state.lead_score,
            "category": self.state.category,
            "reason": self.state.reason,
            "action": str(branch_result)
        })

        with open("leads.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name", "website", "score",
                    "category", "reason", "action"
                ]
            )
            writer.writeheader()
            writer.writerows(self.state.leads)

        print("\nLead saved to leads.csv")

        return "Lead qualification completed"
print("Agent 9 - Lead Qualification Flow is ready!")

# -------------------- AGENT 10: APPROVAL REPORT CREWS --------------------

from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen, or_, router
from crewai.flow.human_feedback import human_feedback
from pydantic import BaseModel
import os


# Reuse the existing llm from Cell 3


# Flow state
class ReportState(BaseModel):
    topic: str = ""
    audience: str = ""
    research_notes: str = ""
    report: str = ""
    feedback: str = ""
    decision: str = ""
    revision_count: int = 0
    max_revisions: int = 3
    final_report: str = ""


# ==========================================
# CREW 1: RESEARCH CREW (3 AGENTS)
# ==========================================

researcher = Agent(
    role="Research Analyst",
    goal="Identify important information about the report topic",
    backstory="An analyst who organizes information into clear findings.",
    llm=llm,
    verbose=False
)

fact_checker = Agent(
    role="Fact Checker",
    goal="Check claims for reliability and identify unsupported facts",
    backstory="A careful reviewer who distinguishes facts from assumptions.",
    llm=llm,
    verbose=False
)

source_organizer = Agent(
    role="Source Organizer",
    goal="Organize source references and research limitations",
    backstory="A research assistant who documents evidence transparently.",
    llm=llm,
    verbose=False
)

research_task = Task(
    description="""
    Research the following topic for the target audience.

    Topic: {topic}
    Audience: {audience}

    Prepare key findings, background, important considerations,
    and practical insights. Clearly distinguish established facts
    from assumptions. Do not invent statistics or citations.
    """,
    expected_output="A structured research brief with key findings.",
    agent=researcher
)

fact_check_task = Task(
    description="""
    Review the research findings provided by the previous task.

    Topic: {topic}
    Audience: {audience}

    Identify unsupported claims, missing context, possible inaccuracies,
    and areas that need verification. Do not claim external verification
    unless a source was actually checked.
    """,
    expected_output="A fact-check review with limitations and corrections.",
    agent=fact_checker,
    context=[research_task]
)

source_task = Task(
    description="""
    Prepare the final research notes for a report writer.

    Topic: {topic}
    Audience: {audience}

    Combine the findings and fact-check review. Include source references
    only when actual URLs or source details are available in the research.
    Otherwise, explicitly list the claims requiring source verification.
    """,
    expected_output="A concise research brief and source-verification notes.",
    agent=source_organizer,
    context=[research_task, fact_check_task]
)

research_crew = Crew(
    agents=[researcher, fact_checker, source_organizer],
    tasks=[research_task, fact_check_task, source_task],
    process=Process.sequential,
    verbose=False
)


# ==========================================
# CREW 2: WRITING CREW
# ==========================================

writer = Agent(
    role="Report Writer",
    goal="Create a clear, professional Markdown report",
    backstory="An experienced technical writer who adapts content to its audience.",
    llm=llm,
    verbose=False
)

writing_task = Task(
    description="""
    Write a complete report in Markdown.

    Topic: {topic}
    Target audience: {audience}
    Research notes: {research_notes}

    Include:
    1. A meaningful title
    2. An executive summary
    3. Main sections with clear headings
    4. Key findings and practical recommendations
    5. Limitations and items requiring verification
    6. Sources, only where actual references are provided

    Do not invent facts, statistics, or references.
    """,
    expected_output="A polished Markdown report ready for human review.",
    agent=writer
)

writing_crew = Crew(
    agents=[writer],
    tasks=[writing_task],
    process=Process.sequential,
    verbose=False
)


# ==========================================
# CREW 3: REVISION CREW
# ==========================================

editor = Agent(
    role="Report Revision Specialist",
    goal="Improve a report using the reviewer's specific feedback",
    backstory="An editor who improves accuracy, clarity, structure, and completeness.",
    llm=llm,
    verbose=False
)

revision_task = Task(
    description="""
    Revise the report using the human reviewer's feedback.

    Topic: {topic}
    Target audience: {audience}
    Previous report:
    {report}

    Human feedback:
    {feedback}

    Address the requested changes while preserving correct information.
    Do not invent facts or citations. Return the COMPLETE revised report
    in Markdown, not merely a list of edits.
    """,
    expected_output="The complete revised Markdown report.",
    agent=editor
)

revision_crew = Crew(
    agents=[editor],
    tasks=[revision_task],
    process=Process.sequential,
    verbose=False
)

# FastAPI application
app = FastAPI(
    title="CrewAI Projects API",
    description="API for running 10 CrewAI projects",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class AgentRequest(BaseModel):
    input: str

class ReviewRequest(BaseModel):
    report_id: str
    feedback: str

@app.get("/")
def home():
    return {
        "message": "CrewAI Projects API is running!",
        "total_agents": 10
    }


# Agent 1 - Blog Post Crew
@app.post("/agent1", tags=["Agent 1 - Blog Post Crew"])
async def run_agent1(request: AgentRequest):
    try:
        result = await blog_crew.kickoff_async(inputs={"topic": request.input})
        return {"output": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent 2 - Competitor Analysis Crew
@app.post("/agent2", tags=["Agent 2 - Competitor Analysis Crew"])
async def run_agent2(request: AgentRequest):
    try:
        urls = [url.strip() for url in request.input.split(",") if url.strip()]
        if len(urls) != 3:
            raise HTTPException(
                status_code=400,
                detail="Please provide exactly 3 competitor URLs separated by commas."
            )
        result = await competitor_crew.kickoff_async(
            inputs={"competitors": ", ".join(urls)}
        )
        return {"output": str(result)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent 3 - Meeting Notes Crew
@app.post("/agent3", tags=["Agent 3 - Meeting Notes Crew"])
async def run_agent3(request: AgentRequest):
    filename = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as file:
            file.write(request.input)
            filename = file.name

        result = await meeting_crew.kickoff_async(inputs={"filename": filename})
        return {"output": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)


# Agent 4 - Data Report Crew
@app.post("/agent4", tags=["Agent 4 - Data Report Crew"])
async def run_agent4(request: AgentRequest):
    filename = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as file:
            file.write(request.input)
            filename = file.name

        result = await data_report_crew.kickoff_async(inputs={"filename": filename})
        return {"output": str(result.raw)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)


# Agent 5 - Customer Support Flow
@app.post("/agent5", tags=["Agent 5 - Customer Support Flow"])
def run_agent5(request: AgentRequest):
    try:
        support_flow = CustomerSupportFlow()
        result = support_flow.kickoff(inputs={"ticket": request.input})
        return {"output": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent 6 - Resume Screening Flow
@app.post("/agent6", tags=["Agent 6 - Resume Screening Flow"])
def run_agent6(request: AgentRequest):
    try:
        resume_flow = ResumeScreeningFlow()
        result = resume_flow.kickoff(inputs={"resume_text": request.input})
        return {"output": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent 7 - Bug Triage Flow
@app.post("/agent7", tags=["Agent 7 - Bug Triage Flow"])
def run_agent7(request: AgentRequest):
    try:
        bug_flow = BugTriageFlow()
        result = bug_flow.kickoff(inputs={"issue_body": request.input})
        return {"output": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent 8 - Content Repurposing Flow
@app.post("/agent8", tags=["Agent 8 - Content Repurposing Flow"])
def run_agent8(request: AgentRequest):
    try:
        content_flow = ContentRepurposingFlow()
        content_flow.kickoff(inputs={"article": request.input})
        return {
            "output": {
                "linkedin_post": content_flow.state.linkedin_post,
                "twitter_thread": content_flow.state.tweet_thread,
                "newsletter": content_flow.state.newsletter,
                "output_files": content_flow.state.output_files
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent 9 - Lead Qualification Flow
@app.post("/agent9", tags=["Agent 9 - Lead Qualification Flow"])
def run_agent9(request: AgentRequest):
    try:
        lead_flow = LeadQualificationFlow()
        lead_flow.kickoff(inputs={"lead_input": request.input})
        return {
            "output": {
                "name": lead_flow.state.name,
                "website": lead_flow.state.website,
                "score": lead_flow.state.lead_score,
                "category": lead_flow.state.category,
                "reason": lead_flow.state.reason,
                "action": lead_flow.state.result,
                "csv_file": "leads.csv"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent 10 - Approval Report Flow
report_sessions = {}

def generate_approval_report(topic, audience):
    if not topic.strip() or not audience.strip():
        raise ValueError("Topic and audience cannot be empty.")

    report_id = str(uuid.uuid4())
    research_result = research_crew.kickoff(inputs={
        "topic": topic,
        "audience": audience
    })
    research_notes = str(research_result)

    writing_result = writing_crew.kickoff(inputs={
        "topic": topic,
        "audience": audience,
        "research_notes": research_notes
    })
    report = str(writing_result)

    report_sessions[report_id] = {
        "topic": topic,
        "audience": audience,
        "report": report,
        "revision_count": 0,
        "max_revisions": 3,
        "decision": "pending"
    }
    return report_id, report


def review_approval_report(report_id, feedback):
    if report_id not in report_sessions:
        raise ValueError("Report ID not found.")

    session = report_sessions[report_id]

    if feedback.strip().lower() == "approved":
        session["decision"] = "approved"
        session["final_report"] = session["report"]
        with open("final.md", "w", encoding="utf-8") as f:
            f.write(session["final_report"])
        return "Report approved and saved as final.md"

    if feedback.strip().lower().startswith("revise"):
        if session["revision_count"] >= session["max_revisions"]:
            session["decision"] = "revision_limit_reached"
            return "Maximum revision limit reached."

        revision_feedback = feedback.split(":", 1)[-1].strip()
        session["revision_count"] += 1
        result = revision_crew.kickoff(inputs={
            "topic": session["topic"],
            "audience": session["audience"],
            "report": session["report"],
            "feedback": revision_feedback
        })
        session["report"] = str(result)
        return session["report"]

    return "Invalid feedback. Enter 'approved' or 'revise: your changes'."


@app.post("/agent10", tags=["Agent 10 - Approval Report Flow"])
def run_agent10(request: AgentRequest):
    try:
        details = request.input.split("|", 1)
        if len(details) != 2:
            raise HTTPException(
                status_code=400,
                detail="Provide input as: topic | audience"
            )
        topic, audience = details[0].strip(), details[1].strip()
        report_id, report = generate_approval_report(topic, audience)
        return {
            "output": report,
            "report_id": report_id,
            "decision": "pending"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent10/review", tags=["Agent 10 - Approval Report Flow"])
def review_agent10(request: ReviewRequest):
    try:
        result = review_approval_report(request.report_id, request.feedback)
        session = report_sessions[request.report_id]
        return {
            "output": result,
            "decision": session["decision"],
            "revision_count": session["revision_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


AGENT_TAGS = [
    {"name": f"Agent {n} - {name}"} for n, name in [
        (1, "Blog Post Crew"),
        (2, "Competitor Analysis Crew"),
        (3, "Meeting Notes Crew"),
        (4, "Data Report Crew"),
        (5, "Customer Support Flow"),
        (6, "Resume Screening Flow"),
        (7, "Bug Triage Flow"),
        (8, "Content Repurposing Flow"),
        (9, "Lead Qualification Flow"),
        (10, "Approval Report Flow")
    ]
]
app.openapi_tags = AGENT_TAGS

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=AGENT_TAGS
    )
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi
app.openapi_schema = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )
