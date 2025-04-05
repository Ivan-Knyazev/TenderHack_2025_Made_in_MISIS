from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(input_password: str, really_password: str) -> bool:
    """Check password"""
    return input_password == really_password
    # return pwd_context.verify(plain_password, hashed_password)


# def get_password_hash(password: str) -> str:
#     """Hash password"""
#     return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any]) -> str:
    """Create JWT"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + \
        timedelta(minutes=int(settings['ACCESS_TOKEN_EXPIRE_MINUTES']))
    # iat - issued at
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    encoded_jwt = jwt.encode(
        to_encode, settings['SECRET_KEY'], algorithm=settings['ALGORITHM'])
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode JWT"""
    try:
        payload = jwt.decode(
            token,
            settings['SECRET_KEY'],
            algorithms=[settings['ALGORITHM']],
            options={"verify_aud": False}
        )
        return payload
    except JWTError as e:
        print(f"JWT Decode Error: {e}")
        return None
