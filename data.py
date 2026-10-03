"""Mock business data for 'Voltra', a fictional European e-scooter rental company.

In production these would be a CRM, a billing system and a help-center search index.
"""

CUSTOMERS = {
    "C-1001": {"name": "Sophie Martin", "tier": "gold", "country": "FR", "language": "French",
               "account_status": "active", "wallet_balance_eur": 12.50},
    "C-1002": {"name": "Jonas Weber", "tier": "standard", "country": "DE", "language": "German",
               "account_status": "active", "wallet_balance_eur": 0.00},
    "C-1003": {"name": "Emma Clarke", "tier": "standard", "country": "UK", "language": "English",
               "account_status": "suspended", "suspension_reason": "3 failed payments",
               "wallet_balance_eur": -18.40},
    "C-1004": {"name": "Daan de Vries", "tier": "platinum", "country": "NL", "language": "Dutch",
               "account_status": "active", "wallet_balance_eur": 44.00},
}

RIDES = {
    "C-1001": [
        {"ride_id": "R-5501", "date": "2026-09-28", "minutes": 14, "cost_eur": 4.90, "status": "completed"},
        {"ride_id": "R-5502", "date": "2026-09-29", "minutes": 312, "cost_eur": 62.40, "status": "completed",
         "note": "Ride not ended in app; scooter parked at 08:52 per GPS"},
    ],
    "C-1002": [
        {"ride_id": "R-6610", "date": "2026-09-27", "minutes": 9, "cost_eur": 3.10, "status": "completed"},
        {"ride_id": "R-6611", "date": "2026-09-27", "minutes": 0, "cost_eur": 3.10, "status": "failed_unlock",
         "note": "Unlock fee charged but scooter never unlocked"},
    ],
    "C-1003": [
        {"ride_id": "R-7001", "date": "2026-09-10", "minutes": 22, "cost_eur": 7.40, "status": "payment_failed"},
    ],
    "C-1004": [
        {"ride_id": "R-8801", "date": "2026-09-29", "minutes": 18, "cost_eur": 5.60, "status": "completed"},
    ],
}

KNOWLEDGE_BASE = [
    {"id": "KB-01", "title": "Refund policy",
     "text": "Agents may refund up to EUR 20 per ride without approval. Refunds above EUR 20 require a "
             "supervisor and must be escalated via a support ticket. Failed unlocks are always refunded in full."},
    {"id": "KB-02", "title": "Forgot to end ride",
     "text": "If GPS shows the scooter was stationary, we recalculate the ride to the time it was parked "
             "plus 5 minutes. The difference is refunded to the original payment method within 5 business days."},
    {"id": "KB-03", "title": "Account suspension",
     "text": "Accounts are suspended after 3 failed payments. To reactivate, the customer must update their "
             "payment method in the app and settle the outstanding balance. Agents cannot lift suspensions manually."},
    {"id": "KB-04", "title": "Scooter won't unlock",
     "text": "Ask the customer to enable Bluetooth, update the app, and try a different scooter. Unlock fees for "
             "failed unlocks are refunded automatically within 24h; if not, agents refund manually."},
    {"id": "KB-05", "title": "Accidents and injuries",
     "text": "Any report of an accident, injury or unsafe vehicle is a safety incident. Do not troubleshoot. "
             "Create a critical ticket for the Safety team immediately and share the emergency number 112."},
    {"id": "KB-06", "title": "Gold and Platinum benefits",
     "text": "Gold members get 10% off rides and priority support. Platinum members get 20% off, free unlocks "
             "and a dedicated support line with a 1 hour response time."},
    {"id": "KB-07", "title": "Data deletion (GDPR)",
     "text": "Customers can request deletion of personal data. Create a ticket for the Privacy team; deletion "
             "is completed within 30 days. Never delete data directly from support tools."},
]
