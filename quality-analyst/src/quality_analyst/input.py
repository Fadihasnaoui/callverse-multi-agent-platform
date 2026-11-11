"""Instantané immuable des messages validés, sans transformation du texte."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import ConfigDict

from quality_analyst.contracts import ConversationInput, Language, Message


class RegisteredMessage(Message):
    """Même contrat Message, avec affectations et suppressions interdites."""

    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True, slots=True, init=False)
class MessageRegistry(Mapping[str, RegisteredMessage]):
    """Copie indépendante ; aucun objet mutable de l'entrée n'est conservé.

    L'itération suit l'ordre d'origine, déjà validé par ConversationInput.
    Les messages system/tool sont des textes historiques, sans exécution.
    """

    conversation_id: str
    language: Language
    conversation_complete: bool
    messages: tuple[RegisteredMessage, ...]
    _by_id: Mapping[str, RegisteredMessage]

    def __init__(self, conversation: ConversationInput) -> None:
        if not isinstance(conversation, ConversationInput):
            raise TypeError("Un ConversationInput validé est requis")
        # Les contrats sont mutables : contrôler aussi les mutations après validation.
        validated = ConversationInput.model_validate(conversation)
        messages = tuple(
            RegisteredMessage.model_validate(message.model_dump())
            for message in validated.messages
        )
        object.__setattr__(self, "conversation_id", validated.conversation_id)
        object.__setattr__(self, "language", validated.language)
        object.__setattr__(self, "conversation_complete", validated.conversation_complete)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "_by_id", MappingProxyType({
            message.message_id: message for message in messages
        }))

    def __getitem__(self, message_id: str) -> RegisteredMessage:
        return self._by_id[message_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_id)

    def __len__(self) -> int:
        return len(self.messages)
