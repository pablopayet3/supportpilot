"""Tools the agent can call. Each one shows up as its own span in LangSmith."""
import itertools
import re

from langchain_core.tools import tool

from data import CUSTOMERS, KNOWLEDGE_BASE, RIDES

REFUND_LIMIT_EUR = 20.0
_ticket_counter = itertools.count(9001)
_refunds_issued: dict[str, float] = {}


@tool
def search_knowledge_base(query: str) -> str:
    """Search Voltra's help-center policies. Use this before answering any policy question."""
    words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for doc in KNOWLEDGE_BASE:
        haystack = set(re.findall(r"\w+", (doc["title"] + " " + doc["text"]).lower()))
        score = len(words & haystack)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda s: s[0], reverse=True)
    if not scored:
        return "No matching policy found."
    return "\n\n".join(f"[{d['id']}] {d['title']}: {d['text']}" for _, d in scored[:2])


@tool
def lookup_customer(customer_id: str) -> str:
    """Get a customer's profile: name, tier, language, account status and wallet balance."""
    customer = CUSTOMERS.get(customer_id.strip().upper())
    if not customer:
        return f"ERROR: no customer found with id {customer_id}"
    return str(customer)


@tool
def get_ride_history(customer_id: str) -> str:
    """List a customer's recent rides with ride_id, duration, cost and status."""
    rides = RIDES.get(customer_id.strip().upper())
    if rides is None:
        return f"ERROR: no rides found for {customer_id}"
    return str(rides)


@tool
def issue_refund(ride_id: str, amount_eur: float, reason: str) -> str:
    """Refund a ride. Amounts above EUR 20 are rejected and must be escalated with create_ticket."""
    all_rides = {r["ride_id"]: r for rides in RIDES.values() for r in rides}
    ride = all_rides.get(ride_id.strip().upper())
    if not ride:
        return f"ERROR: ride {ride_id} does not exist"
    if amount_eur > ride["cost_eur"]:
        return f"ERROR: refund EUR {amount_eur} exceeds ride cost EUR {ride['cost_eur']}"
    if amount_eur > REFUND_LIMIT_EUR:
        return (f"REJECTED: EUR {amount_eur} is above the EUR {REFUND_LIMIT_EUR} agent limit. "
                "Escalate with create_ticket for supervisor approval.")
    if ride_id in _refunds_issued:
        return f"ERROR: ride {ride_id} was already refunded"
    _refunds_issued[ride_id] = amount_eur
    return f"SUCCESS: refunded EUR {amount_eur:.2f} for {ride_id} ({reason})"


@tool
def create_ticket(team: str, priority: str, summary: str) -> str:
    """Escalate to a human team. team: billing | safety | privacy | tech. priority: low | medium | high | critical."""
    ticket_id = f"T-{next(_ticket_counter)}"
    return f"Ticket {ticket_id} created for {team} team with {priority} priority: {summary}"


ALL_TOOLS = [search_knowledge_base, lookup_customer, get_ride_history, issue_refund, create_ticket]
