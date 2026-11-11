# Callverse Multi-Agent Platform

Callverse is an end-to-end multi-agent customer-service platform. It brings together simulated clients, a customer advisor, conversation quality analysis, workforce routing, and marketing operations in one connected workflow.

## End-to-end workflow

1. **Jasmin** creates client personas and simulated customer conversations.
2. **Amira, the Customer Advisor**, handles the travel request through the advisor workflow.
3. **Workforce Manager** routes each request between AI handling and available human advisers, including escalations and reservation requests.
4. **Quality Analyst** evaluates the saved conversation, checks evidence and identifies issues without changing the transcript.
5. **Digital Twin** presents the live simulation, conversations, routing decisions and quality results in a unified dashboard.

## Repository structure

| Folder | Implementation |
| --- | --- |
| `platform/` | Digital Twin platform and shared integration layer |
| `marketing-agent/` | Marketing Agent backend services and dashboard implementation |
| `customer-advisor/` | Amira customer advisor interface, voice listener and scenario library |
| `jasmin-client-simulator/` | Jasmin personas, client simulator and conversation engine |
| `quality-analyst/` | Quality scoring, evidence review and analyst services |
| `workforce-manager/` | AI/human routing, workforce logic and evaluation tests |

Together, these components demonstrate a complete customer-service simulation: a client interacts with Amira, the conversation is routed and monitored, and the result is reviewed for quality.
