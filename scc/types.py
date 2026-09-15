"""All shared data structures. Other modules import from here only.

Plain dataclasses, JSON-serialisable via to_dict()/from_dict().
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

# ----------------------------------------------------------------------------- basic enums / aliases
AtomType = Literal["bool", "categorical", "ordinal_3", "text", "duration"]
# disclosure: plain string, or ("after_n_asks", n)
Disclosure = Any
QuestionForm = Literal["open", "forced_choice", "severity_open", "scale", "yes_no", "point_to", "recall_test", "chat"]
QUESTION_FORMS = ["open", "forced_choice", "severity_open", "scale", "yes_no", "point_to", "recall_test", "chat"]
ChannelKind = Literal["cooperative", "exaggerate", "understate", "vague", "omit", "self_dx", "relay_distort"]
CHANNEL_KINDS = ["cooperative", "exaggerate", "understate", "vague", "omit", "self_dx", "relay_distort"]
SelfDxConviction = Literal["open_mention", "insist", "reframe"]


class ActionKind(str, Enum):
    ASK = "ASK"
    VERIFY_RECORD = "VERIFY_RECORD"
    ASK_FAMILY = "ASK_FAMILY"
    REQUEST_TEST = "REQUEST_TEST"
    DIAGNOSE = "DIAGNOSE"
    CHAT = "CHAT"


class ObsKind(str, Enum):
    PATIENT = "PATIENT"
    RECORD = "RECORD"
    FAMILY = "FAMILY"
    TEST = "TEST"
    END = "END"


# ----------------------------------------------------------------------------- serialisation helpers
def _plain(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _plain(v) for k, v in asdict(obj).items()} if False else {f: _plain(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, tuple) else "|".join(map(str, k)): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def to_dict(obj: Any) -> Any:
    return _plain(obj)


def disclosure_from(x: Any) -> Disclosure:
    """Normalise a disclosure value loaded from JSON/YAML: 'after_n_asks:2' or ['after_n_asks', 2] -> tuple."""
    if isinstance(x, str) and x.startswith("after_n_asks"):
        n = int(x.split(":")[1]) if ":" in x else 2
        return ("after_n_asks", n)
    if isinstance(x, (list, tuple)) and len(x) == 2 and x[0] == "after_n_asks":
        return ("after_n_asks", int(x[1]))
    return x


def disclosure_to_json(d: Disclosure) -> Any:
    return f"after_n_asks:{d[1]}" if isinstance(d, tuple) else d


# ----------------------------------------------------------------------------- case
@dataclass
class Atom:
    id: str
    slot: str
    type: AtomType
    true_value: Any
    options: list | None = None
    disclosure: Disclosure = "on_specific_ask"
    verifiable_by: str | None = None          # record | family | exam | None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.disclosure = disclosure_from(self.disclosure)

    @property
    def tag(self) -> str | None:
        return self.tags[0] if self.tags else None

    def to_dict(self) -> dict:
        d = to_dict(self)
        d["disclosure"] = disclosure_to_json(self.disclosure)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Atom":
        return Atom(id=d["id"], slot=d["slot"], type=d["type"], true_value=d["true_value"], options=d.get("options"),
                    disclosure=d.get("disclosure", "on_specific_ask"), verifiable_by=d.get("verifiable_by"),
                    tags=list(d.get("tags", [])))


@dataclass
class Labels:
    primary_dx: str
    ddx_set: list[str]
    red_flags: list[str] = field(default_factory=list)


@dataclass
class Case:
    case_id: str
    cluster: str
    demographics: dict
    labels: Labels
    atoms: list[Atom]
    exam_and_tests: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    lang_notes: dict = field(default_factory=dict)

    def atom(self, aid: str) -> Atom:
        for a in self.atoms:
            if a.id == aid:
                return a
        raise KeyError(aid)

    def by_slot(self, slot: str) -> list[Atom]:
        return [a for a in self.atoms if a.slot == slot]

    def verifiable(self, by: str) -> list[Atom]:
        return [a for a in self.atoms if a.verifiable_by == by]

    def brief(self, lang: str = "en", chief: str | None = None) -> str:
        d = self.demographics
        age, sex, occ = d.get("age"), d.get("sex", ""), d.get("occupation")
        cc = chief or "chest pain"
        if lang == "zh":
            sexz = {"M": "男", "F": "女"}.get(sex, sex)
            occ_s = f"，{occ}" if occ else ""
            return f"{age} 岁{sexz}性{occ_s}，主诉{cc}。"
        sexe = {"M": "man", "F": "woman"}.get(sex, "patient")
        occ_s = f", {occ}" if occ else ""
        return f"{age}-year-old {sexe}{occ_s}, presenting with {cc}."

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "cluster": self.cluster, "source": self.source, "demographics": self.demographics,
                "labels": to_dict(self.labels), "atoms": [a.to_dict() for a in self.atoms],
                "exam_and_tests": self.exam_and_tests, "lang_notes": self.lang_notes}

    @staticmethod
    def from_dict(d: dict) -> "Case":
        lab = d["labels"]
        return Case(case_id=d["case_id"], cluster=d["cluster"], demographics=d.get("demographics", {}),
                    labels=Labels(lab["primary_dx"], list(lab["ddx_set"]), list(lab.get("red_flags", []))),
                    atoms=[Atom.from_dict(a) for a in d["atoms"]], exam_and_tests=d.get("exam_and_tests", {}),
                    source=d.get("source", {}), lang_notes=d.get("lang_notes", {}))


# ----------------------------------------------------------------------------- patient configuration
@dataclass
class Persona:
    h: int = 2
    e: int = 2
    x: int = 2
    a: int = 2
    c: int = 2
    o: int = 2
    cefr: str = "B"

    def code(self) -> str:
        return f"{self.h}{self.e}{self.x}{self.a}{self.c}{self.o}{self.cefr}"

    @staticmethod
    def from_dict(d: dict) -> "Persona":
        return Persona(**{k: d[k] for k in ("h", "e", "x", "a", "c", "o", "cefr") if k in d})


@dataclass
class ChannelSpec:
    kind: str
    params: dict = field(default_factory=dict)

    def tag(self) -> str:
        if not self.params:
            return self.kind
        return self.kind + "(" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()) if not isinstance(v, dict)) + ")"

    @staticmethod
    def from_dict(d: dict) -> "ChannelSpec":
        return ChannelSpec(kind=d["kind"], params=dict(d.get("params", {})))


@dataclass
class PatientConfig:
    persona: Persona = field(default_factory=Persona)
    channels: list[ChannelSpec] = field(default_factory=list)
    seed: int = 0
    post_verify_shift: int = 0
    sample_params_from_persona: bool = False
    name: str = ""

    def label(self) -> str:
        return self.name or ("+".join(c.tag() for c in self.channels) or "cooperative")

    def kinds(self) -> set[str]:
        return {c.kind for c in self.channels}

    @staticmethod
    def from_dict(d: dict) -> "PatientConfig":
        return PatientConfig(persona=Persona.from_dict(d.get("persona", {})),
                             channels=[ChannelSpec.from_dict(c) for c in d.get("channels", [])],
                             seed=int(d.get("seed", 0)), post_verify_shift=int(d.get("post_verify_shift", 0)),
                             sample_params_from_persona=bool(d.get("sample_params_from_persona", False)),
                             name=d.get("name", ""))


# ----------------------------------------------------------------------------- report table / turn plan
@dataclass
class ReportEntry:
    atom_id: str
    true_value: Any
    report_value: Any
    rule: str
    disclosure: Disclosure
    by_form: dict[str, Any] | None = None     # question_form -> value (vague degrades under forced_choice/point_to)

    def value_for(self, form: str) -> Any:
        if self.by_form and form in self.by_form:
            return self.by_form[form]
        return self.report_value

    def to_dict(self) -> dict:
        d = to_dict(self)
        d["disclosure"] = disclosure_to_json(self.disclosure)
        return d


@dataclass
class ClassifierOutput:
    atom_ids: list[str]
    question_form: str
    is_chat: bool = False
    raw: str = ""


@dataclass
class TurnPlan:
    say: list[tuple[Atom, Any]]
    withheld: list[Atom]
    self_dx: tuple[str, str] | None      # (label, directive)
    is_chat: bool
    form: str
    flooding: list[Atom] = field(default_factory=list)   # associated atoms reported True by flooding (subset of say)


@dataclass
class CheckResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    retry_hint: str = ""
    retries: int = 0
    fallback: bool = False


# ----------------------------------------------------------------------------- doctor <-> env contract
@dataclass
class DoctorReadout:
    topk: dict[str, float]
    confidence: float
    raw: str = ""

    def normalised(self) -> dict[str, float]:
        s = sum(max(0.0, float(v)) for v in self.topk.values())
        if s <= 0:
            return dict(self.topk)
        return {k: max(0.0, float(v)) / s for k, v in self.topk.items()}


@dataclass
class DoctorAction:
    kind: ActionKind
    text: str = ""
    target: str | None = None
    readout: DoctorReadout | None = None

    def to_dict(self) -> dict:
        return to_dict(self)


@dataclass
class Observation:
    kind: ObsKind
    text: str
    turn: int


@dataclass
class DoctorContext:
    case_brief: str
    transcript: list[tuple[str, str]]
    turn: int
    max_turns: int
    ddx_hint: list[str] | None
    actions_available: list[ActionKind]
    lang: str = "en"


@runtime_checkable
class Doctor(Protocol):
    name: str

    def reset(self, ctx: DoctorContext) -> None: ...

    def act(self, ctx: DoctorContext) -> DoctorAction: ...


# ----------------------------------------------------------------------------- per-turn log
@dataclass
class TurnLog:
    episode_id: str
    turn: int
    doctor_action: DoctorAction | None
    question_form: str | None
    hit_atoms: list[str]
    disclosed: list[dict]               # {atom, true_value, report_value, rule, n_asked}
    withheld: list[str]
    self_dx_directive: str | None
    observation: Observation
    checker: dict
    oracle: dict | None = None
    llm_calls: dict = field(default_factory=dict)
    timing_ms: dict = field(default_factory=dict)
    classifier_raw: str = ""

    def to_dict(self) -> dict:
        return to_dict(self)


# ----------------------------------------------------------------------------- component bundle (defined in sim; typed loosely here)
@dataclass
class SimComponents:
    classifier: Any
    phraser: Any
    checker: Any
    role: Any
    name: str = "stub"
