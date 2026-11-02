import logging
import time
from typing import Optional, Tuple

import redis
from django.conf import settings
from django.core.cache import cache
from django.core.cache.backends.base import CacheKeyWarning
from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin

from apps.core.throttling import is_premium_user

logger = logging.getLogger(__name__)


LUA_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])

local current = redis.call("INCR", key)
if current == 1 then
    redis.call("EXPIRE", key, window)
end

local ttl = redis.call("TTL", key)
if current > limit then
    return {0, current, ttl}
else
    return {1, current, ttl}
end
"""

_LUA_SHA = None


def get_redis_client():
    try:
        redis_url = getattr(settings, "REDIS_URL", None) or getattr(
            settings, "RATE_LIMIT_REDIS_URL", None
        )

        if redis_url:
            return redis.from_url(
                redis_url,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
                decode_responses=True,
            )
    except (redis.exceptions.RedisError, ValueError) as e:
        logger.warning("Caught exception: %s", e)
    return None


def _parse_rate(
    rate_setting, default_limit: int, default_window: int = 60
) -> Tuple[int, int]:
    """
    Parse rates like '5/minute', '60/minute', '100/hour' or integer limits into
    (limit, window_seconds).
    """
    if isinstance(rate_setting, int):
        return rate_setting, default_window
    if isinstance(rate_setting, str) and "/" in rate_setting:
        try:
            parts = rate_setting.split("/")
            num = int(parts[0])
            unit = parts[1].lower().strip()
            if unit.startswith("s"):
                duration = 1
            elif unit.startswith("m"):
                duration = 60
            elif unit.startswith("h"):
                duration = 3600
            elif unit.startswith("d"):
                duration = 86400
            else:
                duration = default_window
            return num, duration
        except (ValueError, IndexError, TypeError):
            pass
    elif isinstance(rate_setting, str):
        try:
            return int(rate_setting), default_window
        except ValueError:
            pass
    return default_limit, default_window


def _get_client_ip(request) -> str:
    """
    Extract the real client IP, accounting for trusted proxy hops.
    Never blindly trusts arbitrary X-Forwarded-For headers.
    """
    trusted_hops = getattr(settings, "TRUSTED_PROXY_COUNT", 0)
    xff = request.META.get("HTTP_X_FORWARDED_FOR", "")

    if xff and trusted_hops > 0:
        ips = [ip.strip() for ip in xff.split(",") if ip.strip()]
        real_index = len(ips) - trusted_hops - 1
        if 0 <= real_index < len(ips):
            return ips[real_index]

    return request.META.get("REMOTE_ADDR", "") or "127.0.0.1"


class RateLimitMiddleware(MiddlewareMixin):
    """
    Distributed rate limiting middleware supporting:
    - Endpoint categories (Issue #3061):
      * Auth endpoints (login/signup/refresh/oauth): 5 req/min
      * API reads (GET, HEAD, OPTIONS): 60 req/min
      * API writes (POST, PUT, PATCH, DELETE): 20 req/min
    - Global tier-based hourly budgets:
      * Anonymous: 100 req/hr
      * Authenticated: 1000 req/hr
      * Premium: 10000 req/hr
    Sets standard X-RateLimit-* and Retry-After headers on responses.
    """

    SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
    AUTH_PREFIXES = (
        "/api/auth/",
        "/api/v1/auth/",
        "/oauth/",
        "/api/oauth/",
        "/api/v1/oauth/",
        "/accounts/",
    )

    def __init__(self, get_response):
        super().__init__(get_response)
        # Global tier limits (hourly budgets for backward compatibility)
        tier_anon_str = getattr(settings, "API_RATE_LIMIT_ANON", "100/hour")
        self.anon_limit, _ = _parse_rate(tier_anon_str, 100, 3600)

        tier_auth_str = getattr(settings, "API_RATE_LIMIT_AUTH", "1000/hour")
        self.auth_limit, _ = _parse_rate(tier_auth_str, 1000, 3600)

        tier_prem_str = getattr(settings, "API_RATE_LIMIT_PREMIUM", "10000/hour")
        self.premium_limit, _ = _parse_rate(tier_prem_str, 10000, 3600)

        self.window = getattr(settings, "API_RATE_LIMIT_WINDOW", 3600)

        # Endpoint category limits (Issue #3061)
        auth_rate_setting = getattr(settings, "RATE_LIMIT_AUTH_CATEGORY", "5/minute")
        self.auth_cat_limit, self.auth_cat_window = _parse_rate(
            auth_rate_setting, 5, 60
        )

        read_rate_setting = getattr(settings, "RATE_LIMIT_API_READS", "60/minute")
        self.read_limit, self.read_window = _parse_rate(read_rate_setting, 60, 60)

        write_rate_setting = getattr(settings, "RATE_LIMIT_API_WRITES", "20/minute")
        self.write_limit, self.write_window = _parse_rate(write_rate_setting, 20, 60)

        self.premium_multiplier = getattr(settings, "RATE_LIMIT_PREMIUM_MULTIPLIER", 5)
        self.redis_client = get_redis_client()

    def _get_endpoint_category(self, request) -> str:
        """
        Categorizes incoming request:
        - "auth": Authentication endpoints (login, signup, oauth, etc.)
        - "read": Safe idempotent API methods (GET, HEAD, OPTIONS)
        - "write": Mutating API methods (POST, PUT, PATCH, DELETE)
        """
        path = request.path_info or request.path or ""

        for prefix in self.AUTH_PREFIXES:
            if path.startswith(prefix):
                return "auth"

        if request.method in self.SAFE_METHODS:
            return "read"
        return "write"

    def process_request(self, request):
        if getattr(settings, "DISABLE_RATE_LIMITING", False):
            return None

        path = request.path_info or request.path or ""

        if not path.startswith(("/api/", "/api/v1/", "/oauth/", "/accounts/")):
            return None

        if path.startswith("/api/webhooks/"):
            return None

        if path.startswith(("/admin/", "/static/", "/health/", "/media/")):
            return None

        is_auth = hasattr(request, "user") and request.user.is_authenticated

        if is_auth:
            user_id = getattr(request.user, "pk", getattr(request.user, "id", None))
            is_prem = is_premium_user(request.user)
            if is_prem:
                tier_limit = self.premium_limit
                identifier = f"user:premium:{user_id}"
            else:
                tier_limit = self.auth_limit
                identifier = f"user:{user_id}"
        else:
            is_prem = False
            tier_limit = self.anon_limit
            ip = _get_client_ip(request)
            identifier = f"ip:{ip}"

        category = self._get_endpoint_category(request)
        if category == "auth":
            cat_limit = self.auth_cat_limit
            cat_window = self.auth_cat_window
        elif category == "read":
            cat_limit = self.read_limit * (self.premium_multiplier if is_prem else 1)
            cat_window = self.read_window
        else:
            cat_limit = self.write_limit * (self.premium_multiplier if is_prem else 1)
            cat_window = self.write_window

        # 1. Enforce category limit
        cat_cache_key = f"ratelimit:cat:{category}:{identifier}"
        cat_allowed, cat_remaining, cat_ttl = self._check_rate_limit(
            cat_cache_key, cat_limit, cat_window
        )
        cat_reset_val = cat_ttl if (cat_ttl is not None and cat_ttl > 0) else cat_window

        if not cat_allowed:
            return self._build_429_response(cat_limit, cat_reset_val, category=category)

        # 2. Enforce global tier limit (hourly budget)
        tier_cache_key = f"ratelimit:{identifier}"
        tier_allowed, tier_remaining, tier_ttl = self._check_rate_limit(
            tier_cache_key, tier_limit, self.window
        )
        tier_reset_val = (
            tier_ttl if (tier_ttl is not None and tier_ttl > 0) else self.window
        )

        if not tier_allowed:
            return self._build_429_response(tier_limit, tier_reset_val, category=None)

        # Both allowed: store metadata for response headers
        request._rate_limit_info = {
            "limit": tier_limit,
            "remaining": tier_remaining,
            "reset": tier_reset_val,
            "cat_limit": cat_limit,
            "cat_remaining": cat_remaining,
            "cat_reset": cat_reset_val,
            "category": category,
        }

        return None

    def _build_429_response(
        self, limit: int, reset_val: int, category: Optional[str] = None
    ) -> JsonResponse:
        data = {
            "error": "rate_limited",
            "code": "rate_limited",
            "message": "Request limit exceeded. Please wait before retrying.",
            "retry_after": reset_val,
        }
        response = JsonResponse(data, status=429)
        # Attach .data for test compatibility with DRF APIClient assertions
        response.data = data
        response["Retry-After"] = str(reset_val)
        response["X-RateLimit-Limit"] = str(limit)
        response["X-RateLimit-Remaining"] = "0"
        response["X-RateLimit-Reset"] = str(int(time.time() + reset_val))
        if category:
            response["X-RateLimit-Category"] = category
        return response

    def process_response(self, request, response):
        if hasattr(request, "_rate_limit_info"):
            info = request._rate_limit_info
            reset_secs = info.get("reset") or self.window
            response["X-RateLimit-Limit"] = str(info["limit"])
            response["X-RateLimit-Remaining"] = str(info["remaining"])
            response["X-RateLimit-Reset"] = str(int(time.time() + reset_secs))
            if "category" in info:
                response["X-RateLimit-Category"] = str(info["category"])
                response["X-RateLimit-Category-Limit"] = str(info["cat_limit"])
                response["X-RateLimit-Category-Remaining"] = str(info["cat_remaining"])

        return response

    def _check_rate_limit(self, key, limit, window):
        backend = getattr(settings, "RATE_LIMIT_BACKEND", "local").lower()

        if backend == "redis" and self.redis_client:
            global _LUA_SHA
            try:
                if _LUA_SHA is None:
                    _LUA_SHA = self.redis_client.script_load(LUA_RATE_LIMIT_SCRIPT)

                try:
                    res = self.redis_client.evalsha(_LUA_SHA, 1, key, window, limit)
                except redis.exceptions.NoScriptError:
                    _LUA_SHA = self.redis_client.script_load(LUA_RATE_LIMIT_SCRIPT)
                    res = self.redis_client.evalsha(_LUA_SHA, 1, key, window, limit)

                allowed = bool(res[0])
                current_count = int(res[1])
                ttl = int(res[2]) if res[2] and res[2] > 0 else window
                remaining = max(0, limit - current_count) if allowed else 0

                return allowed, remaining, ttl
            except redis.exceptions.RedisError as e:
                logger.warning(
                    "Rate limit redis lua error, falling back to local/cache: %s", e
                )

        # Fallback to local fixed-window via Django cache
        try:
            current = cache.get(key, 0)
            if current >= limit:
                return False, 0, window

            if current == 0:
                cache.set(key, 1, window)
                return True, limit - 1, window
            else:
                try:
                    current = cache.incr(key)
                except ValueError:
                    current = current + 1
                    cache.set(key, current, window)
                return True, max(0, limit - current), window
        except (
            redis.exceptions.ConnectionError,
            redis.exceptions.TimeoutError,
            CacheKeyWarning,
        ) as e:
            logger.error("Rate limiter failed open: %s", e)
            return True, limit, window
