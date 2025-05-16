from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os
from langsmith import utils
from langsmith import traceable

load_dotenv()



os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multiagent_fucntiondecorator"


utils.tracing_is_enabled()

llm = init_chat_model(
    "openai:gpt-4.1"
)


class MessageClassifier(BaseModel):
    message_type: Literal["legal", "technology", "sales", "marketing", "operations", "hr", "finance", "executive"] = Field(
        ...,
        description="Classify if the message requires an legal or technology or sales or marketing or operations or hr or finance or executive."
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """You are a classification agent in a corporate AI system. Your role is to analyze the user’s message and determine which of the following specialized agents should handle it. Carefully evaluate the intent, language, and domain of the message to make the best decision.

Classify the user message as one of the following categories:

- 'executive': if the query is about business strategy, vision, high-level decision-making, leadership direction, or inter-department coordination.
- 'finance': if the query relates to money, budgeting, expenses, accounting, financial planning, investments, or compliance in financial operations.
- 'hr': if the query involves hiring, employee relations, payroll, workplace issues, HR policies, recruitment, onboarding, or performance management.
- 'operations': if the message focuses on internal process optimization, logistics, supply chain, delivery, workflows, or daily execution tasks.
- 'marketing': if the user asks about promotion, branding, advertising, content creation, customer outreach, social media, or growing audience.
- 'sales': if it involves closing deals, customer leads, CRM, negotiation, pitching, client conversion, or revenue generation.
- 'technology': if the request is about digital infrastructure, IT support, software development, cybersecurity, or system integration.
- 'legal': if it refers to laws, contracts, compliance issues, regulatory requirements, legal risks, or corporate governance.

Return only one of the keywords above based on the user’s intent, like this:
"content": "executive"

Do not include any explanation or extra text—just return the selected keyword exactly.
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}


def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "legal": 
        return {"next": "legal"}
    if message_type == "technology":
        return {"next": "technology"}
    if message_type == "sales":
        return {"next": "sales"}
    if message_type == "marketing": 
        return {"next": "marketing"}
    if message_type == "operations":
        return {"next": "operations"}
    if message_type == "hr":
        return {"next": "hr"}
    if message_type == "finance":
        return {"next": "finance"}
    if message_type == "executive":
        return {"next": "executive"}
    
       

   


    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a compassionate therapist. Focus on the emotional aspects of the user's message.
                        Show empathy, validate their feelings, and help them process their emotions.
                        Ask thoughtful questions to help them explore their feelings more deeply.
                        Avoid giving logical solutions unless explicitly asked."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}



    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a purely logical assistant. Focus only on facts and information.
            Provide clear, concise answers based on logic and evidence.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def executive_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are the Chief Executive Agent responsible for overseeing and aligning the entire organization. You think strategically, set goals, and coordinate cross-department efforts. Your decisions shape the direction, culture, and sustainability of the company. Evaluate risks, ensure mission alignment, and prioritize long-term value creation. Delegate effectively while maintaining high-level insight and control."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def finance_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You manage all financial functions of the organization. This includes budgeting, forecasting, accounting, and financial reporting. Ensure the company is financially healthy, compliant with regulations, and maximizing profitability. You evaluate investments, reduce financial risk, and assist in strategic planning with data-driven insights. Keep track of cash flow and ensure accurate financial statements."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

@traceable(run_type="retriever")
def get_available_positions() -> str:
    """Returns a string listing currently available positions in the company."""
    return "Currently, we have openings for AI Engineers and Data Scientists."

def hr_agent(state: State):
    last_message = state["messages"][-1]
    available_roles = get_available_positions() # HR agent calls the function

    system_prompt = f"""You are responsible for managing the human capital of the company. Oversee recruitment, onboarding, employee engagement, and retention. Design and manage compensation, performance appraisals, and training programs. Ensure legal compliance with labor laws and nurture a healthy workplace culture. Act as a bridge between management and employees.

When asked about available positions, job openings, or recruitment, please use the following information: {available_roles}
"""
    messages = [
        {"role": "system",
         "content": system_prompt
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def operations_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You ensure that the company’s day-to-day functions run smoothly and efficiently. Optimize internal processes, supply chains, resource allocation, and quality control. Identify bottlenecks, reduce operational costs, and enhance workflow productivity. Collaborate across departments to meet delivery timelines and customer expectations. Ensure consistency and scalability in execution."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def marketing_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You lead the company’s marketing initiatives to grow brand awareness and customer engagement. Create and manage campaigns, analyze market trends, and develop positioning strategies. Oversee content creation, advertising, SEO, and performance analytics. Align marketing efforts with sales and product strategies to drive demand. Ensure consistent brand messaging across channels."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def sales_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are responsible for generating revenue and managing customer relationships. Develop sales strategies, close deals, and maintain strong client pipelines. Understand customer needs and align offerings to maximize value and conversion rates. Collaborate with marketing and product teams to refine messaging and offerings. Track KPIs, manage CRM data, and optimize the sales funnel."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def technology_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You manage the organization’s technology infrastructure and digital systems. Ensure network security, maintain uptime, and support users across departments. Oversee software development, system integrations, and tech troubleshooting. Keep systems scalable, secure, and aligned with business goals. Stay up-to-date with emerging technologies to recommend improvements."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def legal_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are the legal advisor responsible for minimizing legal risks and ensuring corporate compliance. Draft, review, and manage contracts, policies, and legal documents. Provide counsel on regulatory requirements, IP protection, and dispute resolution. Ensure ethical governance and risk mitigation across departments. Maintain awareness of local and international laws relevant to the business."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}




graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("executive", executive_agent)
graph_builder.add_node("finance", finance_agent)
graph_builder.add_node("hr", hr_agent)
graph_builder.add_node("operations", operations_agent)
graph_builder.add_node("marketing", marketing_agent)
graph_builder.add_node("sales", sales_agent)
graph_builder.add_node("technology", technology_agent)
graph_builder.add_node("legal", legal_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {"executive": "executive", "finance": "finance" , "hr": "hr", "operations": "operations", "marketing": "marketing", "sales": "sales", "technology": "technology", "legal": "legal"},
)

graph_builder.add_edge("executive", END)
graph_builder.add_edge("finance", END)
graph_builder.add_edge("hr", END)
graph_builder.add_edge("operations", END)
graph_builder.add_edge("marketing", END)
graph_builder.add_edge("sales", END)
graph_builder.add_edge("technology", END)
graph_builder.add_edge("legal", END)

graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Message: ")
        if user_input == "exit":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()
