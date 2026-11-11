"""Explicit operator-reviewed permission AND wire contract required for live access."""
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
import hashlib,ipaddress,socket,re
from urllib.parse import urlsplit,urljoin,unquote


class AccessError(ValueError):pass


@dataclass(frozen=True)
class AccessPolicy:
    status:str='unestablished'
    evidence_file:Path|None=None
    evidence_sha256:str|None=None
    reviewed_by:str|None=None
    expires_at:datetime|None=None
    approved_hosts:tuple=('ozonevoyages.tn',)
    approved_routes:tuple=()
    contract_verified:bool=False

    def require(self,now=None):
        now=now or datetime.now(timezone.utc)
        if self.status!='approved' or not self.reviewed_by or not self.evidence_file or not self.evidence_sha256:
            raise AccessError('LIVE_ACCESS_NOT_AUTHORIZED')
        if not self.expires_at or self.expires_at.tzinfo is None or self.expires_at<=now:raise AccessError('ACCESS_REVIEW_EXPIRED')
        if not self.evidence_file.is_file() or hashlib.sha256(self.evidence_file.read_bytes()).hexdigest()!=self.evidence_sha256:
            raise AccessError('ACCESS_EVIDENCE_MISMATCH')
        if not self.contract_verified or not self.approved_routes:raise AccessError('WIRE_CONTRACT_NOT_VERIFIED')
        if set(self.approved_hosts)-{'ozonevoyages.tn'}:raise AccessError('UNAPPROVED_HOST_CONFIGURATION')


def safe_url(url,policy,*,base='https://ozonevoyages.tn/',method='GET'):
    target=urljoin(base,url);u=urlsplit(target);decoded=unquote(u.path)
    if (u.scheme!='https' or u.hostname not in policy.approved_hosts or u.username or u.password or u.port not in {None,443}
        or u.fragment or '\\' in target or any(ord(c)<32 for c in target+decoded) or '%' in decoded
        or '..' in decoded.split('/') or re.search(r'(^|[?&])(token|password|cookie|authorization|api_key)=',u.query,re.I)):
        raise AccessError('URL_REFUSED')
    if not any(method==verb and (u.path==path or (path.endswith('/') and u.path.startswith(path))) for verb,path in policy.approved_routes):
        raise AccessError('ROUTE_NOT_APPROVED')
    return target


def require_public_dns(host):
    addresses={row[4][0] for row in socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses):raise AccessError('NON_PUBLIC_DNS_REFUSED')
