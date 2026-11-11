"""Résolution exacte des citations LLM ; les positions appartiennent à Python."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from quality_analyst.contracts import ContractModel, Identifier, MessageRole, Text
from quality_analyst.evidence import EvidenceVerifier, VerificationStatus


class QuoteProposal(ContractModel):
    message_id: Identifier
    quote: Annotated[Text, Field(max_length=2000)]
    role: MessageRole


@dataclass(frozen=True, slots=True)
class QuoteResolution:
    message_id: str
    quote: str
    role: MessageRole
    start: int | None
    end: int | None
    status: VerificationStatus
    rejection_code: str | None
    explanation: str
    canonical_evidence_id: str | None = None

    def to_dict(self):
        return {"proposal": {"message_id": self.message_id, "quote": self.quote, "role": self.role.value},
                "start": self.start, "end": self.end, "status": self.status.value,
                "rejection_code": self.rejection_code, "explanation": self.explanation,
                "canonical_evidence_id": self.canonical_evidence_id, "semantic_verified": False}


class QuoteResolver:
    """Un registre canonique via EvidenceVerifier ; aucun ajustement du texte.

    Une répétition entre sections réutilise la même preuve canonique. Une
    répétition dans le message, même chevauchante, reste une ambiguïté.
    """

    def __init__(self, verifier: EvidenceVerifier):
        if not isinstance(verifier, EvidenceVerifier):
            raise TypeError("EvidenceVerifier required")
        self.verifier = verifier
        self._positions: dict[tuple[str, int, int], str] = {}
        self._counter = 0

    def resolve(self, proposal: QuoteProposal) -> QuoteResolution:
        if not isinstance(proposal, QuoteProposal):
            raise TypeError("QuoteProposal required")
        proposal = QuoteProposal.model_validate(proposal)

        def rejected(code, explanation):
            return QuoteResolution(proposal.message_id, proposal.quote, proposal.role,
                                   None, None, VerificationStatus.REJECTED, code, explanation)

        message = self.verifier.registry.get(proposal.message_id)
        if message is None:
            return rejected("MESSAGE_NOT_FOUND", "Message absent du registre ; aucune recherche ailleurs.")
        if message.role != proposal.role:
            return rejected("ROLE_MISMATCH", "Le rôle proposé diffère du rôle canonique.")
        start = message.text.find(proposal.quote)
        if start < 0:
            return rejected("QUOTE_NOT_FOUND", "Aucune occurrence exacte ; aucune normalisation ni réparation.")
        if message.text.find(proposal.quote, start + 1) >= 0:
            return rejected("QUOTE_AMBIGUOUS", "Plusieurs occurrences exactes ; aucune position choisie.")
        end = start + len(proposal.quote)
        position = (message.message_id, start, end)
        evidence_id = self._positions.get(position)
        if evidence_id is None:
            self._counter += 1
            evidence_id = f"E-Q{self._counter:03d}"
        result = self.verifier.verify_span(
            evidence_id=evidence_id, message_id=proposal.message_id,
            start=start, end=end, quote=proposal.quote, expected_role=proposal.role,
        )
        if result.status == VerificationStatus.ACCEPTED:
            self._positions[position] = result.canonical_evidence_id
        return QuoteResolution(proposal.message_id, proposal.quote, proposal.role,
                               start, end, result.status,
                               result.rejection_code.value if result.rejection_code else None,
                               result.explanation, result.canonical_evidence_id)
