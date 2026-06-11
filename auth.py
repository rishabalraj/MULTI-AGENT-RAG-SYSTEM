# api/auth.py
"""
Step 9: Role-Based Access Control (RBAC) & JWT Authentication
Protects API endpoints. Each user has roles that filter which
documents they can access.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict
from functools import wraps

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from loguru import logger

from core.config import settings

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class TokenData(BaseModel):
    user_id: str
    username: str
    roles: List[str] = ["all"]
    department: Optional[str] = None


class UserCredentials(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    roles: List[str]


# ──────────────────────────────────────────────
# Mock User Database (replace with real DB in prod)
# ──────────────────────────────────────────────

# In production, store hashed passwords in your database.
# Format: {username: {"hashed_password": ..., "roles": [...], "user_id": ...}}
MOCK_USERS: Dict[str, Dict] = {
    "admin": {
        "user_id": "usr_001",
        "hashed_password": pwd_context.hash("admin123"),
        "roles": ["all", "admin", "engineering", "hr", "finance"],
        "department": "IT",
    },
    "engineer": {
        "user_id": "usr_002",
        "hashed_password": pwd_context.hash("eng123"),
        "roles": ["all", "engineering"],
        "department": "Engineering",
    },
    "analyst": {
        "user_id": "usr_003",
        "hashed_password": pwd_context.hash("analyst123"),
        "roles": ["all", "finance", "analytics"],
        "department": "Finance",
    },
    "hr_manager": {
        "user_id": "usr_004",
        "hashed_password": pwd_context.hash("hr123"),
        "roles": ["all", "hr"],
        "department": "HR",
    },
}


# ──────────────────────────────────────────────
# Auth Functions
# ──────────────────────────────────────────────

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Verify credentials and return user dict or None."""
    user = MOCK_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return {"username": username, **user}


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT token."""
    payload = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload["exp"] = expire
    payload["iat"] = datetime.utcnow()
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        username: str = payload.get("username")
        roles: List[str] = payload.get("roles", ["all"])

        if not user_id or not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        return TokenData(user_id=user_id, username=username, roles=roles)

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ──────────────────────────────────────────────
# FastAPI Dependencies
# ──────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> TokenData:
    """
    FastAPI dependency: extract & validate user from Bearer token.
    Use as: current_user: TokenData = Depends(get_current_user)
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[TokenData]:
    """
    FastAPI dependency: optional auth. Returns None if no token provided.
    Useful for public endpoints with optional personalization.
    """
    if credentials is None:
        return None
    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Factory for role-checking dependencies.
    Usage: admin_only = require_role("admin")
    """
    async def checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if required_role not in current_user.roles and "admin" not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return current_user
    return checker


# Pre-built role checkers
require_admin = require_role("admin")
require_engineering = require_role("engineering")
