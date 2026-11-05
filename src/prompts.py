"""The test prompts used by both the app and the offline benchmark.

Four prompts come from the fine-tuning task's domain (Azerbaijani Python
tutoring): explaining a concept, writing code, generating a quiz, and checking
a student's answer.  One deliberately tricky prompt sits outside the domain.
"""

from __future__ import annotations

from typing import Iterable

PROMPTS: list[dict] = [
    {
        "id": "explain",
        "title": "1. Python izahı",
        "domain": "in-domain",
        "system": (
            "Sən Azərbaycan dilində Python öyrənən yeni başlayanlara kömək edən "
            "müəllimsən. Sadə və dəqiq izah ver, lazım olanda işlək Python kodu göstər."
        ),
        "user": (
            "Python-da list və tuple arasındakı fərqi yeni başlayan üçün izah et. "
            "Qısa kod nümunəsi ver."
        ),
    },
    {
        "id": "code",
        "title": "2. Kod yazma",
        "domain": "in-domain",
        "system": (
            "Sən Azərbaycan dilində Python öyrənən yeni başlayanlara kömək edən "
            "müəllimsən. Sadə və dəqiq izah ver, lazım olanda işlək Python kodu göstər."
        ),
        "user": (
            "while döngüsü ilə 1-dən 5-ə qədər ədədləri çap edən Python kodu yaz "
            "və hər sətri qısa izah et."
        ),
    },
    {
        "id": "quiz",
        "title": "3. Quiz yaratma",
        "domain": "in-domain",
        "system": (
            "Sən Azərbaycan dilində Python müəllimisən. İstifadəçinin seçdiyi mövzu "
            "üzrə 3 qısa suallıq quiz yarat. Cavab açarını sonda ayrıca göstər."
        ),
        "user": (
            "Python siyahıları mövzusunda 3 suallıq quiz yarat və cavab açarını "
            "sonda göstər."
        ),
    },
    {
        "id": "check",
        "title": "4. Cavabı yoxlama",
        "domain": "in-domain",
        "system": (
            "Sən Azərbaycan dilində Python müəllimisən. Şagirdin cavabını yoxla, "
            "düzgün və səhv hissələri qısa izah et. Səhvdirsə doğru cavabı göstər."
        ),
        "user": "Cavabımı yoxla: `[3, 7, 8][1]` nəticəsi 8-dir. Düzdür?",
    },
    {
        "id": "hard",
        "title": "5. Çətin / domen xarici",
        "domain": "out-of-domain",
        "system": "Sən dəqiq və qısa cavab verən köməkçisən.",
        "user": (
            "Bir marafonda ikinci yerdə gedən idmançını keçdin. İndi neçənci "
            "yerdəsən? Qısa cavab ver və səbəbini bir cümlə ilə izah et."
        ),
    },
]

CUSTOM_PROMPT_ID = "custom"


def get_prompt(prompt_id: str) -> dict:
    for prompt in PROMPTS:
        if prompt["id"] == prompt_id:
            return prompt
    raise KeyError(f"Unknown prompt id: {prompt_id}")


def domain_prompts(prompts: Iterable[dict] | None = None) -> list[dict]:
    source = PROMPTS if prompts is None else prompts
    return [prompt for prompt in source if prompt["domain"] == "in-domain"]


def out_of_domain_prompts(prompts: Iterable[dict] | None = None) -> list[dict]:
    source = PROMPTS if prompts is None else prompts
    return [prompt for prompt in source if prompt["domain"] == "out-of-domain"]
