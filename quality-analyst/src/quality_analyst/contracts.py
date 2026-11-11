"""Contrats métier v0.1.0 ; aucune analyse IA ni aucun calcul de score.

Un rapport isolé est contrôlé structurellement. Le contexte Pydantic optionnel
``conversation`` / ``policy_documents`` permet les contrôles liés aux sources.
Les chargeurs publics rejettent les extensions JSON non standard et doublons.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
import hashlib
import json
import math
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Une chaîne non blanche est requise")
    return value  # Ne pas modifier le texte utilisé par les offsets ou hashes.


def _unique(values: list[str]) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError("Les identifiants référencés doivent être uniques")
    return values


Text = Annotated[str, Field(min_length=1), AfterValidator(_nonblank)]
Identifier = Annotated[str, Field(min_length=1, max_length=128), AfterValidator(_nonblank)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
IndividualScore = Annotated[int, Field(ge=0, le=4)]
Percentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Coverage = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Duration = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifiers = Annotated[list[Identifier], AfterValidator(_unique)]


class Language(StrEnum):
    FR = "fr"
    EN = "en"
    AR = "ar"
    MIXED = "mixed"
    AEB = "aeb"
    UNKNOWN = "unknown"


class MessageRole(StrEnum):
    CUSTOMER = "customer"
    ADVISOR = "advisor"
    SYSTEM = "system"
    TOOL = "tool"
    OTHER = "other"


class MessageChannel(StrEnum):
    TEXT = "text"
    TRANSCRIPT = "transcript"
    UNKNOWN = "unknown"


class AnalysisStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class LanguageSupport(StrEnum):
    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class DimensionStatus(StrEnum):
    EVALUATED = "evaluated"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REJECTED = "rejected"


class ClaimBasis(StrEnum):
    OBSERVABLE = "observable"
    INFERENCE = "inference"
    EXTERNAL_REFERENCE = "external_reference"


class AssertionStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    REQUIRES_EXTERNAL_VERIFICATION = "requires_external_verification"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class FindingSupport(StrEnum):
    ACCEPTED_OBSERVATION = "accepted_observation"
    ACCEPTED_INFERENCE = "accepted_inference"
    REFERENCE_VERIFIED = "reference_verified"


class Severity(StrEnum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class DefectType(StrEnum):
    IRRELEVANT_RESPONSE = "irrelevant_response"
    INCOMPLETE_ANSWER = "incomplete_answer"
    INTERNAL_CONTRADICTION = "internal_contradiction"
    REFERENCE_INCORRECT = "reference_incorrect"
    POLICY_VIOLATION = "policy_violation"
    UNPROFESSIONAL_TONE = "unprofessional_tone"
    MISSED_ACKNOWLEDGEMENT = "missed_acknowledgement"
    UNCLEAR_GUIDANCE = "unclear_guidance"
    UNRESOLVED_NEED = "unresolved_need"
    UNSUPPORTED_GUARANTEE = "unsupported_guarantee"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ResolutionBasis(StrEnum):
    CUSTOMER_CONFIRMATION = "customer_confirmation"
    BUSINESS_RECORD = "business_record"
    ADVISOR_CLAIM = "advisor_claim"
    TRANSCRIPT_ONLY = "transcript_only"
    NONE = "none"


class TrajectorySignal(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TrajectoryBasis(StrEnum):
    SELF_REPORT = "self_report"
    INFERENCE = "inference"


class TrajectoryTrend(StrEnum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ErrorCode(StrEnum):
    INPUT_INVALID = "INPUT_INVALID"
    ROLE_UNCERTAIN = "ROLE_UNCERTAIN"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    OUTPUT_INVALID = "OUTPUT_INVALID"
    EVIDENCE_REJECTED = "EVIDENCE_REJECTED"
    MISSING_APPLICABLE_POLICY = "MISSING_APPLICABLE_POLICY"


class ContractModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class Source(ContractModel):
    """Déclarations de provenance ; null signifie information non fournie."""

    system: Identifier | None
    source_type: Text | None
    provenance_id: Identifier | None
    permitted_use: Text | None


class BusinessContext(ContractModel):
    """Liste blanche de contexte ; pas de champ libre recevant des labels gold."""

    agency_id: Identifier | None
    product_id: Identifier | None
    currency: Annotated[str, Field(pattern=r"^[A-Z]{3}$")] | None
    travel_date: date | None


class Message(ContractModel):
    message_id: Identifier
    sequence: NonNegativeInt
    role: MessageRole
    speaker_id: Identifier | None
    text: Text
    timestamp: AwareDatetime | None
    channel: MessageChannel
    transcription_confidence: Coverage | None


class PolicyReference(ContractModel):
    """Identité versionnée et métadonnées, sans chargement de fichier ni réseau."""

    policy_id: Identifier
    version: Text
    sha256: Sha256
    effective_from: AwareDatetime | None
    effective_to: AwareDatetime | None
    scope: Text | None
    authority: Text | None
    source_uri: Text | None
    provenance_id: Identifier | None
    license_or_permission: Text | None

    @model_validator(mode="after")
    def valid_period(self) -> Self:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to <= self.effective_from
        ):
            raise ValueError("La période de validité doit être croissante")
        return self


class PolicyDocument(PolicyReference):
    text: Text

    @model_validator(mode="after")
    def digest_matches_text(self) -> Self:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("Le SHA-256 ne correspond pas au texte UTF-8 exact")
        return self


class ConversationInput(ContractModel):
    schema_version: Literal["qa.input/0.1.0"]
    conversation_id: Identifier
    language: Language
    messages: Annotated[list[Message], Field(min_length=1)]
    conversation_complete: bool
    source: Source
    occurred_at: AwareDatetime | None
    policy_refs: list[PolicyReference]
    business_context: BusinessContext

    @model_validator(mode="after")
    def unique_ordered_messages(self) -> Self:
        _unique([message.message_id for message in self.messages])
        if any(
            left.sequence >= right.sequence
            for left, right in zip(self.messages, self.messages[1:])
        ):
            raise ValueError("Les séquences des messages doivent être strictement croissantes")
        _unique([policy.policy_id for policy in self.policy_refs])
        return self


class EvidenceSpan(ContractModel):
    evidence_id: Identifier
    message_id: Identifier
    start: NonNegativeInt
    end: NonNegativeInt
    quote: Text
    expected_role: MessageRole
    verification_status: Literal["exact_match"]

    @model_validator(mode="after")
    def valid_span_shape(self) -> Self:
        _span_shape(self.start, self.end, self.quote)
        return self


class GroundedText(ContractModel):
    text: Text
    evidence_ids: Annotated[Identifiers, Field(min_length=1)]


class Need(ContractModel):
    need_id: Identifier
    description: Text
    request_evidence_ids: Annotated[Identifiers, Field(min_length=1)]
    response_evidence_ids: Identifiers
    resolution_status: ResolutionStatus
    resolution_basis: ResolutionBasis
    unresolved_reason: Text | None

    @model_validator(mode="after")
    def resolution_consistency(self) -> Self:
        if self.resolution_status in {
            ResolutionStatus.UNKNOWN,
            ResolutionStatus.UNRESOLVED,
            ResolutionStatus.PARTIALLY_RESOLVED,
        } and self.unresolved_reason is None:
            raise ValueError("Un besoin non résolu ou inconnu nécessite un motif")
        if self.resolution_status == ResolutionStatus.RESOLVED:
            if self.unresolved_reason is not None or not self.response_evidence_ids:
                raise ValueError("Un besoin résolu exige une réponse et aucun motif de non-résolution")
        if self.resolution_status == ResolutionStatus.NOT_APPLICABLE and self.unresolved_reason is not None:
            raise ValueError("Un besoin non applicable n'a pas de motif de non-résolution")
        _resolution_basis(self.resolution_status, self.resolution_basis)
        return self


class Understanding(ContractModel):
    initial_need: GroundedText | None
    needs: list[Need]
    main_issues: list[GroundedText]


class PolicyEvidence(ContractModel):
    policy_id: Identifier
    version: Text
    sha256: Sha256
    start: NonNegativeInt
    end: NonNegativeInt
    quote: Text
    applicability_reason: Text

    @model_validator(mode="after")
    def valid_span_shape(self) -> Self:
        _span_shape(self.start, self.end, self.quote)
        return self


class OmissionEvidence(ContractModel):
    """Demande et fenêtre de réponse, sans conclure par recherche lexicale."""

    need_id: Identifier
    request_evidence_ids: Annotated[Identifiers, Field(min_length=1)]
    response_message_ids: Identifiers
    window_end_message_id: Identifier
    expected_response_kind: Text


class Finding(ContractModel):
    finding_id: Identifier
    defect_type: DefectType
    severity: Severity
    claim_basis: ClaimBasis
    statement: Text
    evidence_ids: Annotated[Identifiers, Field(min_length=1)]
    policy_evidence: list[PolicyEvidence]
    support_status: FindingSupport
    uncertainty: Text | None
    corrected_later: bool
    omission: OmissionEvidence | None

    @model_validator(mode="after")
    def support_consistency(self) -> Self:
        expected = {
            ClaimBasis.OBSERVABLE: FindingSupport.ACCEPTED_OBSERVATION,
            ClaimBasis.INFERENCE: FindingSupport.ACCEPTED_INFERENCE,
            ClaimBasis.EXTERNAL_REFERENCE: FindingSupport.REFERENCE_VERIFIED,
        }
        if self.support_status != expected[self.claim_basis]:
            raise ValueError("Le support doit correspondre à la nature de l'affirmation")
        if self.claim_basis == ClaimBasis.INFERENCE and self.uncertainty is None:
            raise ValueError("Une inférence doit exposer son incertitude")
        if self.claim_basis == ClaimBasis.EXTERNAL_REFERENCE and not self.policy_evidence:
            raise ValueError("Une affirmation externe exige une référence métier")
        if self.defect_type in {
            DefectType.REFERENCE_INCORRECT, DefectType.POLICY_VIOLATION
        } and self.claim_basis != ClaimBasis.EXTERNAL_REFERENCE:
            raise ValueError("Ce défaut exige une référence externe applicable")
        if self.defect_type in {
            DefectType.INCOMPLETE_ANSWER, DefectType.UNRESOLVED_NEED
        } and self.omission is None:
            raise ValueError("Un défaut d'omission exige la fenêtre de réponse")
        if self.omission is not None and self.defect_type not in {
            DefectType.INCOMPLETE_ANSWER,
            DefectType.UNRESOLVED_NEED,
            DefectType.MISSED_ACKNOWLEDGEMENT,
        }:
            raise ValueError("Une fenêtre d'omission doit correspondre au défaut")
        return self


class UnverifiedClaim(ContractModel):
    claim_id: Identifier
    evidence_ids: Annotated[Identifiers, Field(min_length=1)]
    statement: Text
    status: AssertionStatus
    reason: Text
    required_reference: Text | None

    @model_validator(mode="after")
    def uncertainty_only(self) -> Self:
        if self.status not in {
            AssertionStatus.REQUIRES_EXTERNAL_VERIFICATION,
            AssertionStatus.INSUFFICIENT_EVIDENCE,
        }:
            raise ValueError("Une assertion non vérifiée ne peut être soutenue ou contredite")
        if (
            self.status == AssertionStatus.REQUIRES_EXTERNAL_VERIFICATION
            and self.required_reference is None
        ):
            raise ValueError("La référence externe manquante doit être décrite")
        return self


class DimensionAssessment(ContractModel):
    status: DimensionStatus
    score: IndividualScore | None
    rationale: Text
    evidence_ids: Identifiers
    finding_ids: Identifiers
    policy_ids: Identifiers
    uncertainty: Text | None
    rejection_reasons: list[Text]
    rejected_proposal: dict[Text, JsonValue] | None

    @field_validator("rejected_proposal")
    @classmethod
    def finite_proposal(cls, value: dict | None) -> dict | None:
        if value is not None:
            json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def score_matches_status(self) -> Self:
        if self.status == DimensionStatus.EVALUATED:
            if self.score is None or not (self.evidence_ids or self.finding_ids):
                raise ValueError("Une dimension évaluée exige une note et des preuves")
        elif self.score is not None:
            raise ValueError("Une dimension non évaluée doit avoir score=null")
        if self.status == DimensionStatus.REJECTED:
            if not self.rejection_reasons or not self.rejected_proposal:
                raise ValueError("Un rejet exige ses motifs et la proposition conservée")
        elif self.rejection_reasons or self.rejected_proposal is not None:
            raise ValueError("Une dimension acceptée ne doit pas porter de rejet")
        return self


class QualityDimensions(ContractModel):
    relevance: DimensionAssessment
    empathy: DimensionAssessment
    professionalism: DimensionAssessment
    clarity: DimensionAssessment
    resolution: DimensionAssessment
    factual_accuracy: DimensionAssessment
    policy_compliance: DimensionAssessment


class TrajectorySegment(ContractModel):
    start_sequence: NonNegativeInt
    end_sequence: NonNegativeInt
    signal: TrajectorySignal
    claim_basis: TrajectoryBasis
    evidence_ids: Annotated[Identifiers, Field(min_length=1)]
    uncertainty: Text | None

    @model_validator(mode="after")
    def ordered_segment(self) -> Self:
        if self.end_sequence < self.start_sequence:
            raise ValueError("Un segment doit respecter l'ordre de la conversation")
        if (
            self.claim_basis == TrajectoryBasis.INFERENCE
            or self.signal == TrajectorySignal.UNKNOWN
        ) and self.uncertainty is None:
            raise ValueError("Un signal inféré ou inconnu exige une incertitude")
        return self


class CustomerTrajectory(ContractModel):
    segments: list[TrajectorySegment]
    trend: TrajectoryTrend
    uncertainty: Text | None

    @model_validator(mode="after")
    def supported_trend(self) -> Self:
        if self.trend != TrajectoryTrend.UNKNOWN and not self.segments:
            raise ValueError("Une tendance nécessite des segments documentés")
        if self.trend == TrajectoryTrend.UNKNOWN and self.uncertainty is None:
            raise ValueError("Une tendance inconnue nécessite une explication")
        if self.trend in {TrajectoryTrend.IMPROVING, TrajectoryTrend.WORSENING} and len(self.segments) < 2:
            raise ValueError("Une évolution exige au moins deux segments")
        if any(
            left.end_sequence >= right.start_sequence
            for left, right in zip(self.segments, self.segments[1:])
        ):
            raise ValueError("Les segments doivent être chronologiques et sans chevauchement")
        return self


class Resolution(ContractModel):
    status: ResolutionStatus
    basis: ResolutionBasis
    explanation: Text
    evidence_ids: Identifiers
    need_ids: Identifiers

    @model_validator(mode="after")
    def supported_resolution(self) -> Self:
        _resolution_basis(self.status, self.basis)
        if self.basis != ResolutionBasis.NONE and not self.evidence_ids:
            raise ValueError("Une base de résolution déclarée exige une preuve")
        if self.status in {
            ResolutionStatus.RESOLVED, ResolutionStatus.PARTIALLY_RESOLVED, ResolutionStatus.UNRESOLVED
        } and not self.need_ids:
            raise ValueError("Une résolution conclue doit identifier le besoin concerné")
        return self


class Recommendation(ContractModel):
    recommendation_id: Identifier
    finding_ids: Identifiers
    unverified_claim_ids: Identifiers
    action: Text
    priority: Severity
    expected_improvement: Text

    @model_validator(mode="after")
    def linked_action(self) -> Self:
        if not (self.finding_ids or self.unverified_claim_ids):
            raise ValueError("Une recommandation doit être reliée à un constat ou une vérification manquante")
        return self


class Scoring(ContractModel):
    """Valeurs fournies par l'assembleur ; pas de pondération ni de recalcul."""

    partial_score_100: Percentage | None
    global_score_100: Percentage | None
    coverage_absolute: Coverage
    coverage_applicable: Coverage | None
    evaluated_weight: Annotated[int, Field(ge=0, le=100)]
    applicable_weight: Annotated[int, Field(ge=0, le=100)]
    calculation_version: Text
    rubric_version: Text

    @model_validator(mode="after")
    def coherent_availability(self) -> Self:
        if self.evaluated_weight > self.applicable_weight:
            raise ValueError("Le poids évalué dépasse le poids applicable")
        if self.applicable_weight == 0:
            if self.coverage_applicable is not None:
                raise ValueError("Sans dimension applicable, la couverture applicable est null")
        elif self.coverage_applicable is None:
            raise ValueError("Le poids applicable impose une couverture déclarée")
        if self.evaluated_weight == 0:
            if (
                self.partial_score_100 is not None
                or self.global_score_100 is not None
                or self.coverage_absolute != 0
                or self.coverage_applicable not in (None, 0)
            ):
                raise ValueError("Aucun score ni couverture positive sans dimension évaluée")
        elif self.partial_score_100 is None:
            raise ValueError("Une évaluation pondérée exige un score partiel déclaré")
        elif self.coverage_absolute == 0 or self.coverage_applicable == 0:
            raise ValueError("Un poids évalué positif exige une couverture positive")
        if self.applicable_weight > 0 and self.evaluated_weight == self.applicable_weight:
            if self.coverage_applicable != 1:
                raise ValueError("Tous les critères applicables évalués exigent une couverture complète")
        elif self.evaluated_weight < self.applicable_weight and self.coverage_applicable == 1:
            raise ValueError("Une évaluation partielle ne peut déclarer une couverture complète")
        if self.global_score_100 is not None:
            if (
                self.evaluated_weight != self.applicable_weight
                or self.coverage_applicable != 1
                or self.global_score_100 != self.partial_score_100
            ):
                raise ValueError("Un score global exige une couverture complète et le même score partiel")
        return self


class TechnicalError(ContractModel):
    code: ErrorCode
    stage: Text
    message: Text


class ExecutionMetadata(ContractModel):
    package_version: Text
    python_version: Text
    pydantic_version: Text
    rubric_version: Text
    model: Text | None
    model_digest: Sha256 | None
    input_sha256: Sha256 | None
    policy_hashes: dict[Identifier, Sha256]
    configuration: dict[Identifier, JsonValue]
    llm_calls: Annotated[int, Field(ge=0, le=2)]
    tool_calls: NonNegativeInt
    started_at: AwareDatetime | None
    completed_at: AwareDatetime | None
    duration_seconds: Duration | None
    llm_duration_seconds: Duration | None

    @field_validator("configuration")
    @classmethod
    def finite_configuration(cls, value: dict) -> dict:
        json.dumps(value, allow_nan=False)
        return value

    @model_validator(mode="after")
    def coherent_execution(self) -> Self:
        if (self.model is None) != (self.model_digest is None):
            raise ValueError("Le modèle et son digest doivent être fournis ensemble")
        if self.llm_calls > 0 and self.model is None:
            raise ValueError("Des appels LLM exigent l'identité du modèle")
        if self.started_at is None and self.completed_at is not None:
            raise ValueError("Une fin d'exécution exige un début connu")
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("La fin d'exécution précède son début")
        if self.duration_seconds is not None and self.llm_duration_seconds is not None:
            if self.llm_duration_seconds > self.duration_seconds:
                raise ValueError("La durée LLM dépasse la durée totale")
        return self


class QualityReport(ContractModel):
    schema_version: Literal["qa.report/0.1.0"]
    report_id: Identifier
    conversation_id: Identifier
    run_id: Identifier
    analysis_status: AnalysisStatus
    language: Language
    language_support: LanguageSupport
    understanding: Understanding
    evidence: list[EvidenceSpan]
    findings: list[Finding]
    unverified_claims: list[UnverifiedClaim]
    dimensions: QualityDimensions
    customer_trajectory: CustomerTrajectory
    resolution: Resolution
    recommendations: list[Recommendation]
    scoring: Scoring
    limitations: list[Text]
    review_required: bool
    review_reasons: list[Text]
    errors: list[TechnicalError]
    execution: ExecutionMetadata

    @model_validator(mode="after")
    def related_objects(self, info: ValidationInfo) -> Self:
        evidence = _index(self.evidence, "evidence_id")
        positions = {(span.message_id, span.start, span.end) for span in self.evidence}
        if len(positions) != len(self.evidence):
            raise ValueError("Un même passage ne peut devenir plusieurs preuves distinctes")
        needs = _index(self.understanding.needs, "need_id")
        findings = _index(self.findings, "finding_id")
        claims = _index(self.unverified_claims, "claim_id")
        _index(self.recommendations, "recommendation_id")
        grounded = self.understanding.main_issues[:]
        if self.understanding.initial_need is not None:
            grounded.append(self.understanding.initial_need)
        for text in grounded:
            _references(text.evidence_ids, evidence, "preuve")
        if self.understanding.initial_need is not None and not any(
            evidence[key].expected_role == MessageRole.CUSTOMER
            for key in self.understanding.initial_need.evidence_ids
        ):
            raise ValueError("Le besoin initial exige au moins une citation client")
        for need in needs.values():
            _references(need.request_evidence_ids, evidence, "preuve client")
            _references(need.response_evidence_ids, evidence, "preuve de réponse")
            if any(evidence[key].expected_role != MessageRole.CUSTOMER for key in need.request_evidence_ids):
                raise ValueError("Un besoin exige des citations du client")
            if any(evidence[key].expected_role != MessageRole.ADVISOR for key in need.response_evidence_ids):
                raise ValueError("Une réponse au besoin exige des citations du conseiller")
        versions: dict[str, str] = {}
        for finding in findings.values():
            _references(finding.evidence_ids, evidence, "preuve du constat")
            if finding.omission is not None:
                omission = finding.omission
                _references([omission.need_id], needs, "besoin")
                _references(omission.request_evidence_ids, evidence, "preuve de demande")
                if not set(omission.request_evidence_ids) <= set(needs[omission.need_id].request_evidence_ids):
                    raise ValueError("La demande omise ne correspond pas au besoin")
                if not set(omission.request_evidence_ids) <= set(finding.evidence_ids):
                    raise ValueError("Le constat d'omission doit citer sa demande")
            elif not any(evidence[key].expected_role == MessageRole.ADVISOR for key in finding.evidence_ids):
                raise ValueError("Un défaut conseiller exige au moins une citation conseiller")
            if finding.defect_type == DefectType.INTERNAL_CONTRADICTION:
                advisor_spans = [evidence[key] for key in finding.evidence_ids if evidence[key].expected_role == MessageRole.ADVISOR]
                if len(advisor_spans) < 2:
                    raise ValueError("Une contradiction interne exige deux passages conseiller")
            policy_spans = set()
            for policy in finding.policy_evidence:
                _references([policy.policy_id], self.execution.policy_hashes, "politique")
                if policy.sha256 != self.execution.policy_hashes[policy.policy_id]:
                    raise ValueError("Hash de politique incohérent avec l'exécution")
                if versions.setdefault(policy.policy_id, policy.version) != policy.version:
                    raise ValueError("Plusieurs versions d'une politique dans le même rapport")
                key = (policy.policy_id, policy.version, policy.start, policy.end)
                if key in policy_spans:
                    raise ValueError("Passage de politique dupliqué dans un constat")
                policy_spans.add(key)
        for claim in claims.values():
            _references(claim.evidence_ids, evidence, "preuve d'assertion")
            if not any(evidence[key].expected_role == MessageRole.ADVISOR for key in claim.evidence_ids):
                raise ValueError("Une assertion conseiller doit citer le conseiller")
        for name, dimension in self.dimensions:
            _references(dimension.evidence_ids, evidence, "preuve de dimension")
            _references(dimension.finding_ids, findings, "constat")
            _references(dimension.policy_ids, self.execution.policy_hashes, "politique")
            if name == "policy_compliance" and dimension.status == DimensionStatus.EVALUATED and not dimension.policy_ids:
                raise ValueError("La conformité évaluée exige une politique identifiée")
        for segment in self.customer_trajectory.segments:
            _references(segment.evidence_ids, evidence, "preuve de trajectoire")
            if any(evidence[key].expected_role != MessageRole.CUSTOMER for key in segment.evidence_ids):
                raise ValueError("La trajectoire client exige des citations client")
        _references(self.resolution.evidence_ids, evidence, "preuve de résolution")
        _references(self.resolution.need_ids, needs, "besoin de résolution")
        if self.resolution.basis in {
            ResolutionBasis.CUSTOMER_CONFIRMATION, ResolutionBasis.ADVISOR_CLAIM
        }:
            role = (
                MessageRole.CUSTOMER if self.resolution.basis == ResolutionBasis.CUSTOMER_CONFIRMATION
                else MessageRole.ADVISOR
            )
            if not any(evidence[key].expected_role == role for key in self.resolution.evidence_ids):
                raise ValueError("La base de résolution exige une citation du bon auteur")
        for recommendation in self.recommendations:
            _references(recommendation.finding_ids, findings, "constat recommandé")
            _references(recommendation.unverified_claim_ids, claims, "assertion non vérifiée")
        self._status_consistency()
        self._source_consistency(info.context)
        return self

    def _status_consistency(self) -> None:
        dimensions = [dimension for _, dimension in self.dimensions]
        gaps = any(d.status in {DimensionStatus.REJECTED, DimensionStatus.INSUFFICIENT_EVIDENCE} for d in dimensions)
        evaluated = any(d.status == DimensionStatus.EVALUATED for d in dimensions)
        if evaluated != (self.scoring.evaluated_weight > 0):
            raise ValueError("Disponibilité du poids évalué incohérente avec les dimensions")
        if all(d.status == DimensionStatus.NOT_APPLICABLE for d in dimensions) != (self.scoring.applicable_weight == 0):
            raise ValueError("Disponibilité du poids applicable incohérente avec les dimensions")
        if self.scoring.rubric_version != self.execution.rubric_version:
            raise ValueError("Versions de grille incohérentes")
        if self.analysis_status == AnalysisStatus.FAILED and not self.errors:
            raise ValueError("Un échec exige une erreur technique explicite")
        if self.analysis_status == AnalysisStatus.INSUFFICIENT_EVIDENCE and not self.limitations:
            raise ValueError("Les preuves insuffisantes doivent être expliquées")
        if self.analysis_status == AnalysisStatus.COMPLETED and (self.errors or gaps or self.unverified_claims):
            raise ValueError("Un rapport avec lacunes ou erreurs ne peut être completed")
        if self.scoring.global_score_100 is not None and (
            self.analysis_status != AnalysisStatus.COMPLETED or gaps or self.unverified_claims or self.errors
        ):
            raise ValueError("Le score global est indisponible sur un rapport incomplet")
        if self.language == Language.UNKNOWN and self.language_support == LanguageSupport.VALIDATED:
            raise ValueError("Une langue inconnue ne peut être déclarée validée")
        requires_review = bool(
            self.errors or gaps or self.unverified_claims
            or self.analysis_status in {AnalysisStatus.INSUFFICIENT_EVIDENCE, AnalysisStatus.FAILED}
            or self.language_support != LanguageSupport.VALIDATED
            or any(f.severity in {Severity.MAJOR, Severity.CRITICAL} for f in self.findings)
        )
        if requires_review and not self.review_required:
            raise ValueError("Les risques ou incertitudes déclarés imposent une revue")
        if self.review_required != bool(self.review_reasons):
            raise ValueError("Le besoin de revue et ses motifs doivent être cohérents")

    def _source_consistency(self, context: Any) -> None:
        if context is None:
            return
        if not isinstance(context, dict):
            raise ValueError("Le contexte de validation doit être un dictionnaire")
        conversation = context.get("conversation")
        if conversation is not None:
            if not isinstance(conversation, ConversationInput):
                raise ValueError("Le contexte conversation doit être un ConversationInput validé")
            if conversation.conversation_id != self.conversation_id or conversation.language != self.language:
                raise ValueError("Le rapport ne correspond pas à la conversation")
            messages = {message.message_id: message for message in conversation.messages}
            for span in self.evidence:
                _references([span.message_id], messages, "message")
                message = messages[span.message_id]
                if message.role != span.expected_role or message.text[span.start:span.end] != span.quote:
                    raise ValueError("Citation exacte ou auteur incompatible avec le message source")
                if span.end > len(message.text):
                    raise ValueError("Le span dépasse le texte du message")
            spans = {span.evidence_id: span for span in self.evidence}
            for segment in self.customer_trajectory.segments:
                for key in segment.evidence_ids:
                    sequence = messages[spans[key].message_id].sequence
                    if not segment.start_sequence <= sequence <= segment.end_sequence:
                        raise ValueError("Une citation client est hors du segment temporel")
            for finding in self.findings:
                if finding.omission is not None:
                    omission = finding.omission
                    _references(omission.response_message_ids + [omission.window_end_message_id], messages, "message de fenêtre")
                    request_end = max(messages[spans[key].message_id].sequence for key in omission.request_evidence_ids)
                    window_end = messages[omission.window_end_message_id].sequence
                    if window_end < request_end or any(
                        not request_end < messages[key].sequence <= window_end
                        or messages[key].role != MessageRole.ADVISOR
                        for key in omission.response_message_ids
                    ):
                        raise ValueError("Fenêtre de réponse incohérente avec la demande")
            refs = {reference.policy_id: reference for reference in conversation.policy_refs}
            for key, digest in self.execution.policy_hashes.items():
                _references([key], refs, "politique autorisée dans l'entrée")
                if refs[key].sha256 != digest:
                    raise ValueError("Hash de politique différent de la référence d'entrée")
            for finding in self.findings:
                for policy in finding.policy_evidence:
                    if refs[policy.policy_id].version != policy.version:
                        raise ValueError("Version de politique différente de la référence d'entrée")
            if not conversation.conversation_complete:
                if self.scoring.global_score_100 is not None:
                    raise ValueError("Une conversation incomplète interdit le score global")
                if self.resolution.status == ResolutionStatus.UNRESOLVED or any(
                    need.resolution_status == ResolutionStatus.UNRESOLVED for need in self.understanding.needs
                ):
                    raise ValueError("Une fin absente ne démontre pas un besoin définitivement non résolu")
                if not self.review_required:
                    raise ValueError("Une conversation incomplète nécessite une revue")
        documents = context.get("policy_documents")
        if documents is not None:
            if not isinstance(documents, (list, tuple)) or any(not isinstance(doc, PolicyDocument) for doc in documents):
                raise ValueError("policy_documents doit contenir des PolicyDocument validés")
            policies = _index(documents, "policy_id")
            for key, digest in self.execution.policy_hashes.items():
                _references([key], policies, "document de politique")
                if policies[key].sha256 != digest:
                    raise ValueError("Hash incompatible avec le document de politique")
            for finding in self.findings:
                for span in finding.policy_evidence:
                    _references([span.policy_id], policies, "document de politique")
                    document = policies[span.policy_id]
                    if document.version != span.version or document.sha256 != span.sha256:
                        raise ValueError("Version ou hash de document incompatible avec la preuve")
                    if document.text[span.start:span.end] != span.quote or span.end > len(document.text):
                        raise ValueError("Citation exacte absente du document métier")
            if conversation is not None:
                for key in self.execution.policy_hashes:
                    reference = next(ref for ref in conversation.policy_refs if ref.policy_id == key)
                    if policies[key].version != reference.version:
                        raise ValueError("Version de document différente de la référence d'entrée")
            # La portée commerciale et le sens de la règle restent hors de cette validation.


def _span_shape(start: int, end: int, quote: str) -> None:
    if end <= start or end - start != len(quote):
        raise ValueError("Offsets non vides et longueur Unicode exacte de la citation requis")


def _resolution_basis(status: ResolutionStatus, basis: ResolutionBasis) -> None:
    if status in {ResolutionStatus.RESOLVED, ResolutionStatus.PARTIALLY_RESOLVED, ResolutionStatus.UNRESOLVED} and basis == ResolutionBasis.NONE:
        raise ValueError("Une résolution conclue doit préciser sa base")
    if status == ResolutionStatus.NOT_APPLICABLE and basis != ResolutionBasis.NONE:
        raise ValueError("Une résolution non applicable n'a pas de base factuelle")


def _index(items: Any, attribute: str) -> dict:
    keys = [getattr(item, attribute) for item in items]
    _unique(keys)
    return dict(zip(keys, items))


def _references(keys: list[str], available: Any, kind: str) -> None:
    missing = set(keys) - set(available)
    if missing:
        raise ValueError(f"Référence {kind} inexistante : {', '.join(sorted(missing))}")


def strict_json_loads(raw: str | bytes | bytearray) -> Any:
    """Décodage local strict, sans modifier json.loads ni ses valeurs par défaut."""

    def pairs(values: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Clé JSON dupliquée : {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ValueError(f"Nombre JSON non fini interdit : {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nombre JSON non fini interdit : {value}")
        return number

    if not isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("Le chargeur strict attend du texte JSON ou des octets")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)


def load_conversation_json(raw: str | bytes | bytearray) -> ConversationInput:
    strict_json_loads(raw)
    return ConversationInput.model_validate_json(raw, strict=True)


def load_quality_report_json(
    raw: str | bytes | bytearray,
    *,
    conversation: ConversationInput | None = None,
    policy_documents: list[PolicyDocument] | None = None,
) -> QualityReport:
    strict_json_loads(raw)
    return QualityReport.model_validate_json(
        raw,
        strict=True,
        context={"conversation": conversation, "policy_documents": policy_documents},
    )
