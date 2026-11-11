# Ozone Client Simulator

The simulator calls the public API only:

```text
POST /v1/agents/customer-advisor/respond
```

It must use a stable `customer_id`, one `session_id` per conversation, and a
unique `message_id` per submitted message. It must never import `search_api.py`
or any internal Python function.

`tmp-callverse-pages/` contains visual reference PNGs, not simulator source.
