############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# auth.py: API authentication and authorization middleware
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API authentication and authorization."""

from typing import Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from itsdangerous import URLSafeTimedSerializer

from backend.app.db import crud
from backend.app.db.models import ApiKey, User, UserRole, Group
from backend.app.db.session import get_async_db
from backend.app.security.api_keys import api_key_rejection_reason, verify_api_key
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

# Security scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_api_key_from_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """
    Extract API key from request.

    Supports:
    - Authorization: Bearer <key>
    - X-API-Key: <key>
    """
    # Try Authorization header first
    if credentials and credentials.credentials:
        return credentials.credentials

    # Try X-API-Key header
    x_api_key = request.headers.get("X-API-Key")
    if x_api_key:
        return x_api_key

    return None


async def authenticate_request(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    api_key_str: Optional[str] = Depends(get_api_key_from_request),
) -> Tuple[User, ApiKey]:
    """
    Authenticate a request using API key.

    Args:
        request: The incoming request
        db: Database session
        api_key_str: API key from request

    Returns:
        Tuple of (User, ApiKey)

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key_str:
        logger.warning("missing_api_key", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via 'Authorization: Bearer <key>' or 'X-API-Key: <key>'",
        )

    # Verify API key
    api_key = await verify_api_key(db, api_key_str)

    if not api_key:
        logger.warning(
            "invalid_api_key",
            path=request.url.path,
            key_prefix=api_key_str[:8] if len(api_key_str) >= 8 else "short",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Post-verify gate — one source of truth (security/api_keys.py) shared
    # with every other verify_api_key caller (MCP SSE, admin-or-session
    # wrappers, dashboard tts-voices) so status/expiry/user checks cannot
    # drift between call sites.
    rejection = api_key_rejection_reason(api_key)
    if rejection is not None:
        if rejection == "API key has expired":
            logger.warning("expired_api_key", key_id=api_key.id)
        elif rejection == "User account is inactive":
            user = api_key.user
            logger.warning(
                "inactive_user",
                key_id=api_key.id,
                user_id=user.id if user else None,
            )
        else:
            logger.warning(
                "inactive_api_key",
                key_id=api_key.id,
                status=api_key.status.value,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=rejection,
        )

    # NOTE: api_key usage update moved to request completion phase
    # to avoid holding a row lock for the entire request duration.

    return api_key.user, api_key


async def require_role(
    required_role: UserRole,
    user: User = Depends(lambda u=Depends(authenticate_request): u[0]),
) -> User:
    """
    Require a minimum role level.

    Role hierarchy: admin > faculty > staff > student
    """
    role_hierarchy = {
        UserRole.STUDENT: 0,
        UserRole.STAFF: 1,
        UserRole.FACULTY: 2,
        UserRole.ADMIN: 3,
    }

    user_level = role_hierarchy.get(user.role, 0)
    required_level = role_hierarchy.get(required_role, 0)

    if user_level < required_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires {required_role.value} role or higher",
        )

    return user


def _deny_unscoped(api_key, scope: str) -> None:
    """Reject a key whose scope list does not permit ``scope``.

    Scopes intersect with the owner's group-derived privilege; they never
    extend it. So a key minted by a registered app is not an admin credential
    even when its owner is an administrator — administrators use first-party
    apps too, and an app-minted key must not carry their admin rights.
    """
    from backend.app.security.scopes import key_has_scope

    if not key_has_scope(api_key, scope):
        logger.warning(
            "api_key_scope_denied",
            key_id=getattr(api_key, "id", None),
            required_scope=scope,
            app_id=getattr(api_key, "app_id", None),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This credential is not permitted to perform that action",
        )


def require_admin():
    """Dependency that requires admin role (via group.is_admin).

    A scoped key must additionally carry the `admin` scope; group membership
    alone is no longer sufficient for keys that declare a scope list.
    """
    async def check_admin(
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> User:
        from backend.app.security.scopes import SCOPE_ADMIN

        user, api_key = auth_result
        if not user.group or not user.group.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )
        _deny_unscoped(api_key, SCOPE_ADMIN)
        return user
    return check_admin


def require_admin_read():
    """Dependency that requires admin or auditor role (read-only admin access)."""
    async def check_admin_read(
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> User:
        from backend.app.security.scopes import SCOPE_ADMIN

        user, api_key = auth_result
        if not user.group or not user.group.has_admin_read:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin or auditor access required",
            )
        _deny_unscoped(api_key, SCOPE_ADMIN)
        return user
    return check_admin_read


def require_scope(scope: str):
    """Dependency requiring a key to carry an explicit scope.

    Unlike the admin dependencies this is not a narrowing of group privilege —
    it gates capabilities that no group confers, such as an app provisioning
    users on its own behalf. A legacy key (NULL scopes) does NOT satisfy it:
    these capabilities are opt-in by construction, so an old broad key can
    never drift into them.
    """
    async def check_scope(
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> Tuple[User, ApiKey]:
        from backend.app.security.scopes import is_scoped, key_has_scope

        user, api_key = auth_result
        if not is_scoped(api_key) or not key_has_scope(api_key, scope):
            logger.warning(
                "api_key_scope_denied",
                key_id=getattr(api_key, "id", None),
                required_scope=scope,
                app_id=getattr(api_key, "app_id", None),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This credential is not permitted to perform that action",
            )
        return user, api_key
    return check_scope


def require_admin_or_session():
    """
    Dependency that requires admin role via API key OR session cookie.

    This enables admin-only API endpoints to be called from both:
    - Programmatic clients (via API key in Authorization/X-API-Key header)
    - Dashboard AJAX calls (via session cookie from browser login)
    """
    async def check_admin(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
        db: AsyncSession = Depends(get_async_db),
    ) -> User:
        # Try API key auth first
        api_key_str = None
        if credentials and credentials.credentials:
            api_key_str = credentials.credentials
        if not api_key_str:
            api_key_str = request.headers.get("X-API-Key")

        if api_key_str:
            # API key path — delegate to standard auth
            from backend.app.security.api_keys import verify_api_key as _verify
            api_key = await _verify(db, api_key_str)
            # Shared post-verify gate: rejects revoked, expired, and
            # inactive/deleted-user keys — same checks as authenticate_request
            if api_key and api_key_rejection_reason(api_key) is None:
                from backend.app.security.scopes import SCOPE_ADMIN, key_has_scope

                user = api_key.user
                # Same rule as require_admin: a scoped key must also carry the
                # admin scope. Without this check here, an app-minted key
                # belonging to an administrator would still reach every admin
                # endpoint that uses this dependency.
                if (
                    user.group
                    and user.group.is_admin
                    and key_has_scope(api_key, SCOPE_ADMIN)
                ):
                    return user
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key or insufficient permissions",
            )

        # Fallback to session cookie (signed with itsdangerous)
        session_data = request.cookies.get("mindrouter_session")
        if session_data:
            try:
                settings = get_settings()
                serializer = URLSafeTimedSerializer(settings.secret_key, salt="session")
                user_id = int(serializer.loads(session_data, max_age=86400 * 7))
                user = await crud.get_user_by_id(db, user_id)
                if user and user.is_active and user.group and user.group.is_admin:
                    return user
            except Exception:
                pass

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return check_admin


def require_admin_read_or_session():
    """
    Dependency that requires admin or auditor role via API key OR session cookie.

    Like require_admin_or_session but also allows auditor groups (read-only admin).
    """
    async def check_admin_read(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
        db: AsyncSession = Depends(get_async_db),
    ) -> User:
        # Try API key auth first
        api_key_str = None
        if credentials and credentials.credentials:
            api_key_str = credentials.credentials
        if not api_key_str:
            api_key_str = request.headers.get("X-API-Key")

        if api_key_str:
            from backend.app.security.api_keys import verify_api_key as _verify
            api_key = await _verify(db, api_key_str)
            # Shared post-verify gate: rejects revoked, expired, and
            # inactive/deleted-user keys — same checks as authenticate_request
            if api_key and api_key_rejection_reason(api_key) is None:
                from backend.app.security.scopes import SCOPE_ADMIN, key_has_scope

                user = api_key.user
                # A scoped key must also carry the admin scope — see
                # require_admin. Session cookies below carry no scope, so the
                # check applies only on the API-key path.
                if (
                    user.group
                    and user.group.has_admin_read
                    and key_has_scope(api_key, SCOPE_ADMIN)
                ):
                    return user
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key or insufficient permissions",
            )

        # Fallback to session cookie
        session_data = request.cookies.get("mindrouter_session")
        if session_data:
            try:
                settings = get_settings()
                serializer = URLSafeTimedSerializer(settings.secret_key, salt="session")
                user_id = int(serializer.loads(session_data, max_age=86400 * 7))
                user = await crud.get_user_by_id(db, user_id)
                if user and user.is_active and user.group and user.group.has_admin_read:
                    return user
            except Exception:
                pass

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return check_admin_read


class AuthenticatedUser:
    """Dependency class for getting authenticated user."""

    def __init__(self, require_admin: bool = False, require_role: Optional[UserRole] = None):
        self.require_admin_flag = require_admin
        self.require_role = require_role

    async def __call__(
        self,
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> User:
        user, _api_key = auth_result

        if self.require_admin_flag:
            from backend.app.security.scopes import SCOPE_ADMIN

            if not user.group or not user.group.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required",
                )
            _deny_unscoped(_api_key, SCOPE_ADMIN)
        elif self.require_role:
            # Legacy role hierarchy check (kept for backward compat)
            role_hierarchy = {
                UserRole.STUDENT: 0,
                UserRole.STAFF: 1,
                UserRole.FACULTY: 2,
                UserRole.ADMIN: 3,
            }
            user_level = role_hierarchy.get(user.role, 0)
            required_level = role_hierarchy.get(self.require_role, 0)

            if user_level < required_level:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires {self.require_role.value} role or higher",
                )

        return user


class AuthenticatedApiKey:
    """Dependency class for getting authenticated API key."""

    async def __call__(
        self,
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> ApiKey:
        _, api_key = auth_result
        return api_key


# Convenience dependencies
get_current_user = AuthenticatedUser()
get_current_user_admin = AuthenticatedUser(require_admin=True)
get_current_api_key = AuthenticatedApiKey()
