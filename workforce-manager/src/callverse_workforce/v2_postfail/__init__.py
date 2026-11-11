"""V2 post-AI-failure decision context (docs/v2_headroom_spec.md). V1.1 code paths are never modified.

The V2 Twin is a subclass of the frozen V1.1 DigitalTwin: only the post-visible-failure consequence, the human
queue discipline (HIGH class + aging), the V2 WAIT counter mask and the handoff-once rule differ.
"""
