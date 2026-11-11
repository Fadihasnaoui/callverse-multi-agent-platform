from dataclasses import dataclass
import hashlib
@dataclass(frozen=True)
class Response:
    url:str
    body:bytes
    content_type:str
    fetched_at:object
    @property
    def sha256(self):return hashlib.sha256(self.body).hexdigest()
