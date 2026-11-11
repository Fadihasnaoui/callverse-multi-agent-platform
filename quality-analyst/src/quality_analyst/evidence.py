"""Vérification déterministe des preuves ; aucune conclusion sémantique.

Les résultats sont des diagnostics opérationnels, pas de nouveaux contrats
métier. Seuls des EvidenceSpan existants sont exportés vers le futur rapport.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import ConfigDict, TypeAdapter, ValidationError

from quality_analyst.contracts import (
    DefectType,
    EvidenceSpan,
    Finding,
    Identifier,
    MessageRole,
    Need,
    OmissionEvidence,
    ResolutionStatus,
    Text,
)
from quality_analyst.input import MessageRegistry


class VerificationStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RejectionCode(StrEnum):
    INVALID_EVIDENCE_ID = "INVALID_EVIDENCE_ID"
    INVALID_MESSAGE_ID = "INVALID_MESSAGE_ID"
    MESSAGE_NOT_FOUND = "MESSAGE_NOT_FOUND"
    INVALID_OFFSETS = "INVALID_OFFSETS"
    NEGATIVE_OFFSET = "NEGATIVE_OFFSET"
    INVALID_SPAN_ORDER = "INVALID_SPAN_ORDER"
    SPAN_OUT_OF_BOUNDS = "SPAN_OUT_OF_BOUNDS"
    INVALID_QUOTE = "INVALID_QUOTE"
    QUOTE_MISMATCH = "QUOTE_MISMATCH"
    INVALID_EXPECTED_ROLE = "INVALID_EXPECTED_ROLE"
    ROLE_MISMATCH = "ROLE_MISMATCH"
    EVIDENCE_ID_CONFLICT = "EVIDENCE_ID_CONFLICT"
    DUPLICATE_EVIDENCE = "DUPLICATE_EVIDENCE"
    UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE"
    INVALID_RELATION_INPUT = "INVALID_RELATION_INPUT"
    ADVISOR_EVIDENCE_REQUIRED = "ADVISOR_EVIDENCE_REQUIRED"
    DISTINCT_ADVISOR_SPANS_REQUIRED = "DISTINCT_ADVISOR_SPANS_REQUIRED"
    NEED_MISMATCH = "NEED_MISMATCH"
    CUSTOMER_REQUEST_REQUIRED = "CUSTOMER_REQUEST_REQUIRED"
    INVALID_RESPONSE_WINDOW = "INVALID_RESPONSE_WINDOW"
    INCOMPLETE_RESPONSE_WINDOW = "INCOMPLETE_RESPONSE_WINDOW"
    INCOMPLETE_CONVERSATION = "INCOMPLETE_CONVERSATION"
    CORRECTION_NOT_LATER = "CORRECTION_NOT_LATER"


@dataclass(frozen=True, slots=True)
class EvidenceVerificationResult:
    status: VerificationStatus
    rejection_code: RejectionCode | None
    explanation: str
    evidence_id: object
    message_id: object
    start: object
    end: object
    quote: object
    expected_role: object
    evidence: EvidenceSpan | None = None
    canonical_evidence_id: str | None = None
    semantic_verified: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class RelationVerificationResult:
    status: VerificationStatus
    rejection_code: RejectionCode | None
    explanation: str
    relation: str
    subject_id: str | None
    evidence_ids: tuple[str, ...] = ()
    message_ids: tuple[str, ...] = ()
    review_required: bool = True
    conversation_complete: bool | None = None
    window_reaches_end: bool | None = None
    semantic_verified: bool = field(default=False, init=False)
    proves_unresolved_need: bool = field(default=False, init=False)


_IDENTIFIER = TypeAdapter(Identifier, config=ConfigDict(strict=True))
_TEXT = TypeAdapter(Text, config=ConfigDict(strict=True))


class EvidenceVerifier:
    """Un registre, un ensemble canonique de preuves et un journal par instance.

    Les appels sont séquentiels. Les retours et exports sont des copies : ils ne
    permettent pas de modifier les preuves utilisées par les contrôles suivants.
    """

    def __init__(self, registry: MessageRegistry) -> None:
        if not isinstance(registry, MessageRegistry):
            raise TypeError("Un MessageRegistry est requis")
        self._registry = registry
        self._accepted: dict[str, EvidenceSpan] = {}
        self._positions: dict[tuple[str, int, int], str] = {}
        self._checks: list[EvidenceVerificationResult] = []
        self._relation_checks: list[RelationVerificationResult] = []

    @property
    def registry(self) -> MessageRegistry:
        return self._registry

    @property
    def accepted_evidence(self) -> tuple[EvidenceSpan, ...]:
        """Uniquement les preuves exactes, une fois par position physique."""
        return tuple(span.model_copy(deep=True) for span in self._accepted.values())

    @property
    def checks(self) -> tuple[EvidenceVerificationResult, ...]:
        return deepcopy(tuple(self._checks))

    @property
    def relation_checks(self) -> tuple[RelationVerificationResult, ...]:
        return deepcopy(tuple(self._relation_checks))

    def verify(self, evidence: EvidenceSpan) -> EvidenceVerificationResult:
        """Revérifie une EvidenceSpan ; exact_match déclaré n'est pas une preuve."""
        if not isinstance(evidence, EvidenceSpan):
            raise TypeError("Une EvidenceSpan est requise ; sinon utiliser verify_span")
        return self.verify_span(
            evidence_id=evidence.evidence_id, message_id=evidence.message_id,
            start=evidence.start, end=evidence.end, quote=evidence.quote,
            expected_role=evidence.expected_role,
        )

    def verify_span(
        self, *, evidence_id: object, message_id: object, start: object,
        end: object, quote: object, expected_role: object,
    ) -> EvidenceVerificationResult:
        """Vérifie les propositions brutes sans coercition ni réparation.

        Les champs bruts permettent de journaliser aussi des offsets invalides
        qui seraient refusés avant construction d'une EvidenceSpan Pydantic.
        Un seul rejet est retourné : le premier dans l'ordre des contrôles.
        """
        def finish(
            code: RejectionCode | None, explanation: str,
            span: EvidenceSpan | None = None, canonical: str | None = None,
        ) -> EvidenceVerificationResult:
            result = EvidenceVerificationResult(
                status=VerificationStatus.REJECTED if code else VerificationStatus.ACCEPTED,
                rejection_code=code, explanation=explanation,
                evidence_id=evidence_id, message_id=message_id, start=start, end=end,
                quote=quote, expected_role=expected_role, evidence=span,
                canonical_evidence_id=canonical,
            )
            self._checks.append(deepcopy(result))
            return result

        for value, code in (
            (evidence_id, RejectionCode.INVALID_EVIDENCE_ID),
            (message_id, RejectionCode.INVALID_MESSAGE_ID),
        ):
            try:
                _IDENTIFIER.validate_python(value)
            except ValidationError:
                return finish(code, "Identifiant non conforme au contrat Identifier.")
        message = self.registry.get(message_id)
        if message is None:
            return finish(RejectionCode.MESSAGE_NOT_FOUND, "Message absent du registre ; aucune recherche ailleurs.")
        if type(start) is not int or type(end) is not int:
            return finish(RejectionCode.INVALID_OFFSETS, "Offsets entiers stricts requis ; bool, texte et flottants refusés.")
        if start < 0 or end < 0:
            return finish(RejectionCode.NEGATIVE_OFFSET, "Les offsets négatifs sont interdits.")
        if start >= end:
            return finish(RejectionCode.INVALID_SPAN_ORDER, "La fenêtre doit respecter start < end.")
        if end > len(message.text):
            return finish(RejectionCode.SPAN_OUT_OF_BOUNDS, "La fin dépasse la longueur Unicode Python du message.")
        try:
            _TEXT.validate_python(quote)
        except ValidationError:
            return finish(RejectionCode.INVALID_QUOTE, "Citation non conforme au contrat Text.")
        if message.text[start:end] != quote:
            return finish(RejectionCode.QUOTE_MISMATCH, "Le segment exact ne correspond pas à la citation ; aucun ajustement.")
        try:
            if not isinstance(expected_role, (str, MessageRole)):
                raise ValueError
            role = MessageRole(expected_role)
        except ValueError:
            return finish(RejectionCode.INVALID_EXPECTED_ROLE, "Rôle inconnu ; aucune normalisation du rôle.")
        if message.role != role:
            return finish(RejectionCode.ROLE_MISMATCH, "Le rôle attendu diffère du rôle réel du message.")

        # L'auteur exporté est fixé à partir du registre, après comparaison.
        span = EvidenceSpan(
            evidence_id=evidence_id, message_id=message.message_id,
            start=start, end=end, quote=quote, expected_role=message.role,
            verification_status="exact_match",
        )
        key = (message.message_id, start, end)
        previous = self._accepted.get(evidence_id)
        if previous is not None:
            if previous != span:
                return finish(RejectionCode.EVIDENCE_ID_CONFLICT, "Cet identifiant désigne déjà une autre preuve.")
            return finish(None, "Preuve déjà vérifiée ; nouvel export inutile.", previous.model_copy(deep=True), evidence_id)
        if key in self._positions:
            return finish(RejectionCode.DUPLICATE_EVIDENCE, "Cette position possède déjà un identifiant canonique.", canonical=self._positions[key])
        self._accepted[evidence_id] = span.model_copy(deep=True)
        self._positions[key] = evidence_id
        return finish(None, "Citation et attribution exactes ; pertinence non évaluée.", span, evidence_id)

    def _relation(
        self, relation: str, subject_id: str | None, code: RejectionCode | None,
        explanation: str, *, evidence_ids: tuple[str, ...] = (),
        message_ids: tuple[str, ...] = (), window_reaches_end: bool | None = None,
    ) -> RelationVerificationResult:
        result = RelationVerificationResult(
            status=VerificationStatus.REJECTED if code else VerificationStatus.ACCEPTED,
            rejection_code=code, explanation=explanation, relation=relation,
            subject_id=subject_id, evidence_ids=evidence_ids, message_ids=message_ids,
            conversation_complete=self.registry.conversation_complete,
            window_reaches_end=window_reaches_end,
        )
        self._relation_checks.append(deepcopy(result))
        return result

    def verify_finding(
        self, finding: Finding, *, need: Need | None = None,
    ) -> RelationVerificationResult:
        """Prérequis d'un constat candidat ; ne valide jamais sa conclusion."""
        try:
            if not isinstance(finding, Finding):
                raise TypeError
            finding = Finding.model_validate(finding)
        except (ValidationError, TypeError):
            return self._relation("finding", None, RejectionCode.INVALID_RELATION_INPUT, "Un Finding conforme est requis.")
        ids = tuple(finding.evidence_ids)

        def finish(code: RejectionCode | None, explanation: str) -> RelationVerificationResult:
            return self._relation("finding", finding.finding_id, code, explanation, evidence_ids=ids)

        if any(evidence_id not in self._accepted for evidence_id in ids):
            return finish(RejectionCode.UNKNOWN_EVIDENCE, "Toutes les preuves doivent avoir été acceptées par cette instance.")
        advisors = [self._accepted[evidence_id] for evidence_id in ids
                    if self.registry[self._accepted[evidence_id].message_id].role == MessageRole.ADVISOR]
        if finding.omission is not None:
            if need is None or not set(finding.omission.request_evidence_ids).issubset(ids):
                return finish(RejectionCode.NEED_MISMATCH, "Le besoin et ses preuves de demande doivent être fournis et liés au constat.")
            omission_check = self.verify_omission(need, finding.omission)
            if omission_check.status == VerificationStatus.REJECTED:
                return finish(omission_check.rejection_code, omission_check.explanation)
            if finding.defect_type == DefectType.UNRESOLVED_NEED:
                if not self.registry.conversation_complete:
                    return finish(RejectionCode.INCOMPLETE_CONVERSATION, "Capture interrompue : non-résolution définitive refusée.")
                if not omission_check.window_reaches_end:
                    return finish(RejectionCode.INCOMPLETE_RESPONSE_WINDOW, "Une non-résolution finale exige une fenêtre allant jusqu'à la fin capturée.")
            # Une omission peut n'avoir aucune réponse : sa preuve est la fenêtre.
            return finish(None, "Fenêtre d'omission contrôlée ; absence de réponse satisfaisante non démontrée.")
        if not advisors:
            return finish(RejectionCode.ADVISOR_EVIDENCE_REQUIRED, "Un défaut attribué au conseiller exige une citation réelle du conseiller.")
        if finding.defect_type == DefectType.INTERNAL_CONTRADICTION:
            distinct = any(
                left.message_id != right.message_id
                or left.end <= right.start or right.end <= left.start
                for i, left in enumerate(advisors) for right in advisors[i + 1:]
            )
            if not distinct:
                return finish(RejectionCode.DISTINCT_ADVISOR_SPANS_REQUIRED, "Deux passages conseiller distincts et non chevauchants sont requis.")
        return finish(None, "Attribution et références contrôlées ; défaut sémantique non vérifié.")

    def verify_omission(
        self, need: Need, omission: OmissionEvidence,
    ) -> RelationVerificationResult:
        """Contrôle une fenêtre exhaustive de réponses ; jamais leur adéquation."""
        try:
            if not isinstance(need, Need) or not isinstance(omission, OmissionEvidence):
                raise TypeError
            need = Need.model_validate(need)
            omission = OmissionEvidence.model_validate(omission)
        except (ValidationError, TypeError):
            return self._relation("omission", None, RejectionCode.INVALID_RELATION_INPUT, "Need et OmissionEvidence conformes requis.")
        ids = tuple(omission.request_evidence_ids)
        message_ids = tuple(omission.response_message_ids) + (omission.window_end_message_id,)

        def finish(
            code: RejectionCode | None, explanation: str, reaches_end: bool | None = None,
        ) -> RelationVerificationResult:
            return self._relation("omission", need.need_id, code, explanation,
                                  evidence_ids=ids, message_ids=message_ids,
                                  window_reaches_end=reaches_end)

        if omission.need_id != need.need_id or not set(ids).issubset(need.request_evidence_ids):
            return finish(RejectionCode.NEED_MISMATCH, "Identité du besoin ou références de demande incompatibles.")
        if any(evidence_id not in self._accepted for evidence_id in (
            *need.request_evidence_ids, *need.response_evidence_ids,
        )):
            return finish(RejectionCode.UNKNOWN_EVIDENCE, "Les preuves du besoin doivent être acceptées dans le même registre.")
        requests = [self._accepted[evidence_id] for evidence_id in need.request_evidence_ids]
        if any(self.registry[span.message_id].role != MessageRole.CUSTOMER for span in requests):
            return finish(RejectionCode.CUSTOMER_REQUEST_REQUIRED, "La demande doit citer le client réel.")
        if any(message_id not in self.registry for message_id in message_ids):
            return finish(RejectionCode.MESSAGE_NOT_FOUND, "Un message de la fenêtre est absent du registre.")
        request_end = max(self.registry[self._accepted[evidence_id].message_id].sequence
                          for evidence_id in ids)
        window_end = self.registry[omission.window_end_message_id]
        if window_end.sequence < request_end:
            return finish(RejectionCode.INVALID_RESPONSE_WINDOW, "La fin de fenêtre précède la demande.")
        replies = [self.registry[message_id] for message_id in omission.response_message_ids]
        if any(message.role != MessageRole.ADVISOR
               or not request_end < message.sequence <= window_end.sequence for message in replies):
            return finish(RejectionCode.INVALID_RESPONSE_WINDOW, "Les réponses doivent être des messages conseiller situés après la demande et dans la fenêtre.")
        if any(left.sequence >= right.sequence for left, right in zip(replies, replies[1:])):
            return finish(RejectionCode.INVALID_RESPONSE_WINDOW, "Les réponses inspectées doivent suivre l'ordre du registre.")
        expected = tuple(message.message_id for message in self.registry.messages
                         if message.role == MessageRole.ADVISOR
                         and request_end < message.sequence <= window_end.sequence)
        if tuple(omission.response_message_ids) != expected:
            return finish(RejectionCode.INCOMPLETE_RESPONSE_WINDOW, "La liste doit inclure toutes les réponses conseiller de la fenêtre déclarée.")
        reaches_end = window_end.message_id == self.registry.messages[-1].message_id
        if need.resolution_status == ResolutionStatus.UNRESOLVED:
            if not self.registry.conversation_complete:
                return finish(RejectionCode.INCOMPLETE_CONVERSATION, "Une capture incomplète ne justifie pas un besoin définitivement non résolu.", reaches_end)
            if not reaches_end:
                return finish(RejectionCode.INCOMPLETE_RESPONSE_WINDOW, "La fenêtre ne couvre pas la fin capturée pour ce besoin déclaré non résolu.", reaches_end)
        return finish(None, "Fenêtre et attribution contrôlées ; sens de la réponse et résolution non évalués.", reaches_end)

    def verify_later_correction(
        self, original_evidence_id: str, correction_evidence_id: str,
    ) -> RelationVerificationResult:
        """Relie deux passages conseiller dans l'ordre, sans effacer le premier."""
        ids = (original_evidence_id, correction_evidence_id)

        def finish(code: RejectionCode | None, explanation: str) -> RelationVerificationResult:
            return self._relation("later_correction", None, code, explanation, evidence_ids=ids)

        try:
            for evidence_id in ids:
                _IDENTIFIER.validate_python(evidence_id)
        except ValidationError:
            return finish(RejectionCode.INVALID_RELATION_INPUT, "Deux identifiants de preuve conformes sont requis.")
        if any(evidence_id not in self._accepted for evidence_id in ids):
            return finish(RejectionCode.UNKNOWN_EVIDENCE, "Les deux passages doivent avoir été acceptés.")
        original, correction = (self._accepted[evidence_id] for evidence_id in ids)
        first, later = (self.registry[span.message_id] for span in (original, correction))
        if first.role != MessageRole.ADVISOR or later.role != MessageRole.ADVISOR:
            return finish(RejectionCode.ADVISOR_EVIDENCE_REQUIRED, "L'affirmation et sa correction candidate doivent provenir du conseiller.")
        if later.sequence < first.sequence or (
            later.sequence == first.sequence and correction.start < original.end
        ):
            return finish(RejectionCode.CORRECTION_NOT_LATER, "La correction doit suivre l'affirmation : message ultérieur ou segment disjoint ultérieur du même message.")
        return finish(None, "Ordre vérifié ; affirmation initiale conservée, effet correctif non évalué.")
