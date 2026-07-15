"""Application settings and environment loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)
load_dotenv()


def _get_env_str(name: str, default: str = "") -> str:
	return os.getenv(name, default).strip()


def _get_env_int(name: str, default: int) -> int:
	raw_value = os.getenv(name)
	if raw_value is None or raw_value.strip() == "":
		return default
	return int(raw_value)


@dataclass(frozen=True, slots=True)
class AppSettings:
	"""Typed application settings sourced from environment variables."""

	project_root: Path
	app_title: str
	app_env: str
	log_level: str
	max_upload_size_mb: int
	gemini_api_key: str
	gemini_model: str
	embedding_model: str
	pinecone_index_dimension: int
	pinecone_metric: str
	pinecone_api_key: str
	pinecone_index_name: str
	pinecone_namespace: str
	pinecone_cloud: str
	pinecone_region: str
	reports_dir: Path
	assets_dir: Path

	@property
	def max_upload_size_bytes(self) -> int:
		return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
	"""Load and cache application settings once per process."""

	return AppSettings(
		project_root=PROJECT_ROOT,
		app_title=_get_env_str("APP_TITLE", "AI Log Intelligence Platform"),
		app_env=_get_env_str("APP_ENV", "development"),
		log_level=_get_env_str("LOG_LEVEL", "INFO"),
		max_upload_size_mb=_get_env_int("MAX_UPLOAD_SIZE_MB", 200),
		gemini_api_key=_get_env_str("GEMINI_API_KEY"),
		gemini_model=_get_env_str("GEMINI_MODEL", "gemini-2.5-flash"),
		embedding_model=_get_env_str("EMBEDDING_MODEL", "gemini-embedding-2"),
		pinecone_index_dimension=_get_env_int("PINECONE_INDEX_DIMENSION", 3072),
		pinecone_metric=_get_env_str("PINECONE_METRIC", "cosine"),
		pinecone_api_key=_get_env_str("PINECONE_API_KEY"),
		pinecone_index_name=_get_env_str("PINECONE_INDEX_NAME"),
		pinecone_namespace=_get_env_str("PINECONE_NAMESPACE"),
		pinecone_cloud=_get_env_str("PINECONE_CLOUD"),
		pinecone_region=_get_env_str("PINECONE_REGION"),
		reports_dir=PROJECT_ROOT / "reports",
		assets_dir=PROJECT_ROOT / "assets",
	)
