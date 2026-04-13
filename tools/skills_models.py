from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import re

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

class SecretCollection(BaseModel):
    env_var: str
    prompt: Optional[str] = None
    provider_url: Optional[str] = Field(None, alias="url")
    secret: bool = True
    
    @field_validator("env_var", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return str(v).strip()

class SkillSetup(BaseModel):
    help: Optional[str] = None
    collect_secrets: List[SecretCollection] = Field(default_factory=list)

class RequiredEnvVar(BaseModel):
    name: str = Field(alias="env_var")
    prompt: Optional[str] = None
    help: Optional[str] = Field(None, alias="provider_url")
    required_for: Optional[str] = None
    
    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: Any) -> str:
        name = str(v).strip()
        if not _ENV_VAR_NAME_RE.match(name):
            raise ValueError(f"Invalid environment variable name: {name}")
        return name

class SkillPrerequisites(BaseModel):
    env_vars: List[str] = Field(default_factory=list)
    commands: List[str] = Field(default_factory=list)
    
    @field_validator("env_vars", "commands", mode="before")
    @classmethod
    def normalize_list(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            return [v]
        return [str(item) for item in v if str(item).strip()]

class SkillMetadataHermes(BaseModel):
    tags: List[str] = Field(default_factory=list)
    related_skills: List[str] = Field(default_factory=list)

class SkillMetadata(BaseModel):
    hermes: Optional[SkillMetadataHermes] = None
    model_config = {"extra": "allow"}

class SkillFrontmatter(BaseModel):
    name: str = Field(..., max_length=64)
    description: Optional[str] = Field(default="", max_length=1024)
    version: Optional[str] = None
    license: Optional[str] = None
    platforms: Optional[List[str]] = None
    compatibility: Optional[str] = None
    
    tags: List[str] = Field(default_factory=list)
    prerequisites: Optional[SkillPrerequisites] = None
    setup: Optional[SkillSetup] = None
    required_environment_variables: List[Union[str, RequiredEnvVar, Dict[str, Any]]] = Field(default_factory=list)
    metadata: Optional[SkillMetadata] = None

    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, list):
            return [str(t).strip() for t in v if t]
        tags_str = str(v).strip()
        if tags_str.startswith("[") and tags_str.endswith("]"):
            tags_str = tags_str[1:-1]
        return [t.strip().strip("\"'") for t in tags_str.split(",") if t.strip()]

    model_config = {"extra": "allow", "populate_by_name": True}

class SkillMeta(BaseModel):
    """Minimal metadata returned by search results."""
    name: str
    description: str
    source: str
    identifier: str
    trust_level: str
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

class SkillBundle(BaseModel):
    """A downloaded skill ready for quarantine/scanning/installation."""
    name: str
    files: Dict[str, Union[str, bytes]]
    source: str
    identifier: str
    trust_level: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
