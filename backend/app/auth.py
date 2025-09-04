import os
from typing import Optional

from fastapi import Header, HTTPException
import jwt
import json
from pathlib import Path


def _require_bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return authorization.split(" ", 1)[1].strip()


def require_write_auth(authorization: Optional[str] = Header(default=None)):
    # Read config at call time to honor runtime changes (tests, hot reloads)
    api_token = os.getenv("API_TOKEN")
    oidc_hs256_secret = os.getenv("OIDC_HS256_SECRET")
    oidc_issuer = os.getenv("OIDC_ISSUER")
    oidc_audience = os.getenv("OIDC_AUDIENCE")
    oidc_jwks_url = os.getenv("OIDC_JWKS_URL")
    oidc_jwks_inline = os.getenv("OIDC_JWKS")
    oidc_jwks_path = os.getenv("OIDC_JWKS_PATH")

    # If neither method configured, allow dev writes with admin role
    if not (oidc_hs256_secret or api_token or oidc_jwks_url or oidc_jwks_inline or oidc_jwks_path):
        return {"auth": "dev-mode", "roles": ["admin"]}

    # Require a Bearer token header if auth is configured
    token = _require_bearer(authorization)

    # Decide which validator to use
    # If a static API token is configured and the presented token doesn't look like a JWT, use static token path
    if api_token and token.count('.') < 2:
        if token != api_token:
            raise HTTPException(status_code=403, detail="Invalid token")
        role = os.getenv("API_ROLE", "admin").strip().lower()
        if role not in {"viewer", "editor", "admin"}:
            role = "admin"
        return {"auth": "static-token", "roles": [role]}

    treat_as_jwt = False
    header = {}
    try:
        header = jwt.get_unverified_header(token)
    except Exception:
        header = {}
    if (oidc_jwks_url or oidc_jwks_inline or oidc_jwks_path) and (header.get("alg", "").upper().startswith("RS") or not oidc_hs256_secret):
        treat_as_jwt = True
    elif oidc_hs256_secret and not (oidc_jwks_url or oidc_jwks_inline or oidc_jwks_path):
        treat_as_jwt = True
    elif oidc_hs256_secret and (oidc_jwks_url or oidc_jwks_inline or oidc_jwks_path):
        alg = header.get("alg", "").upper()
        treat_as_jwt = alg.startswith("RS") or alg.startswith("HS")

    if treat_as_jwt:
        # RS256 with JWKS
        if (oidc_jwks_inline or oidc_jwks_path or oidc_jwks_url) and header.get("alg", "").upper().startswith("RS"):
            try:
                if oidc_jwks_inline:
                    jwks = json.loads(oidc_jwks_inline)
                elif oidc_jwks_path:
                    jwks = json.loads(Path(oidc_jwks_path).read_text(encoding="utf-8"))
                else:
                    from jwt import PyJWKClient
                    jwk_client = PyJWKClient(oidc_jwks_url)  # type: ignore
                    signing_key = jwk_client.get_signing_key_from_jwt(token)
                    key = signing_key.key
                    options = {"verify_signature": True, "verify_exp": True, "verify_aud": bool(oidc_audience)}
                    claims = jwt.decode(
                        token,
                        key,
                        algorithms=["RS256"],
                        audience=oidc_audience if oidc_audience else None,
                        issuer=oidc_issuer if oidc_issuer else None,
                        options=options,
                    )
                    claims = dict(claims)
                    roles = _extract_roles(claims)
                    claims["roles"] = list(roles)
                    return claims

                keys = jwks.get("keys", []) if isinstance(jwks, dict) else jwks
                kid = header.get("kid")
                selected = None
                for k in keys:
                    if not kid or k.get("kid") == kid:
                        selected = k
                        if kid:
                            break
                if not selected:
                    raise HTTPException(status_code=403, detail="No matching JWK found")
                from jwt.algorithms import RSAAlgorithm
                key = RSAAlgorithm.from_jwk(json.dumps(selected))
                options = {"verify_signature": True, "verify_exp": True, "verify_aud": bool(oidc_audience)}
                claims = jwt.decode(
                    token,
                    key,
                    algorithms=["RS256"],
                    audience=oidc_audience if oidc_audience else None,
                    issuer=oidc_issuer if oidc_issuer else None,
                    options=options,
                )
                claims = dict(claims)
                roles = _extract_roles(claims)
                claims["roles"] = list(roles)
                return claims
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=401, detail="Token expired")
            except Exception as e:
                raise HTTPException(status_code=403, detail=f"Invalid RS256 token: {str(e)}")
        try:
            options = {"verify_signature": True, "verify_exp": True, "verify_aud": bool(oidc_audience)}
            claims = jwt.decode(
                token,
                oidc_hs256_secret,
                algorithms=["HS256"],
                audience=oidc_audience if oidc_audience else None,
                issuer=oidc_issuer if oidc_issuer else None,
                options=options,
            )
            claims = dict(claims)
            roles = _extract_roles(claims)
            claims["roles"] = list(roles)
            return claims
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=403, detail=f"Invalid token: {str(e)}")

    if api_token:
        if token != api_token:
            raise HTTPException(status_code=403, detail="Invalid token")
        role = os.getenv("API_ROLE", "admin").strip().lower()
        if role not in {"viewer", "editor", "admin"}:
            role = "admin"
        return {"auth": "static-token", "roles": [role]}

    # If we reach here, auth configured but no method matched
    raise HTTPException(status_code=401, detail="Unauthorized")


# --- RBAC helpers ---
ROLE_ORDER = {"viewer": 0, "editor": 1, "admin": 2}


def role_required(min_role: str):
    min_role = min_role.lower()
    if min_role not in ROLE_ORDER:
        raise ValueError("Unknown role")

    def _dep(authorization: Optional[str] = Header(default=None)):
        # Chain to write auth then check roles using the Authorization header
        auth_claims = require_write_auth(authorization)  # re-use parsing logic
        roles = set([r.lower() for r in auth_claims.get("roles", [])])
        has = any(ROLE_ORDER.get(r, -1) >= ROLE_ORDER[min_role] for r in roles)
        if not has:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return auth_claims

    return _dep


def _extract_roles(claims: dict) -> set:
    roles = set()
    for key in ("roles", "role", "scope", "permissions"):
        if key in claims and claims[key]:
            val = claims[key]
            if isinstance(val, str):
                roles.update([p.strip().lower() for p in val.split() if p.strip()])
            elif isinstance(val, (list, tuple)):
                roles.update([str(p).strip().lower() for p in val])
    if not roles:
        roles = {"viewer"}
    return roles
