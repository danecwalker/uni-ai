from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

from companion_ai.emotion.detector import EmotionObservation


@dataclass(frozen=True)
class EmotionSignalSettings:
    window: int = 12
    min_non_neutral: int = 8
    min_avg_confidence: float = 0.35
    spike_confidence: float = 0.65
    spike_frames: int = 4


@dataclass(frozen=True)
class EmotionSignalSummary:
    dominant_emotion: str
    average_confidence: float
    max_confidence: float
    non_neutral_frames: int
    total_frames: int
    distribution: dict[str, int]
    sustained_threshold_met: bool
    spike_threshold_met: bool

    @property
    def threshold_met(self) -> bool:
        return self.sustained_threshold_met or self.spike_threshold_met

    @property
    def distribution_text(self) -> str:
        return " ".join(f"{emotion}={count}" for emotion, count in self.distribution.items())


class RollingEmotionSignal:
    def __init__(self, settings: EmotionSignalSettings):
        self.settings = settings
        self._observations: Deque[EmotionObservation] = deque(maxlen=settings.window)

    def add(self, observation: EmotionObservation) -> EmotionSignalSummary:
        self._observations.append(observation)
        return self.summary()

    def clear(self) -> None:
        self._observations.clear()

    def summary(self) -> EmotionSignalSummary:
        non_neutral = [
            observation
            for observation in self._observations
            if observation.label != "neutral"
        ]
        distribution = _distribution(non_neutral)
        confidences = [observation.confidence for observation in non_neutral]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0
        sustained_threshold_met = (
            len(self._observations) >= self.settings.window
            and len(non_neutral) >= self.settings.min_non_neutral
            and average_confidence >= self.settings.min_avg_confidence
        )
        spike_threshold_met = (
            sum(
                1
                for observation in non_neutral
                if observation.confidence >= self.settings.spike_confidence
            )
            >= self.settings.spike_frames
        )
        return EmotionSignalSummary(
            dominant_emotion=_dominant_emotion(non_neutral),
            average_confidence=average_confidence,
            max_confidence=max_confidence,
            non_neutral_frames=len(non_neutral),
            total_frames=len(self._observations),
            distribution=distribution,
            sustained_threshold_met=sustained_threshold_met,
            spike_threshold_met=spike_threshold_met,
        )


class IdleDecisionCooldown:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last_declined_at: Optional[float] = None

    def can_consider(self, now: float) -> bool:
        return (
            self._last_declined_at is None
            or now - self._last_declined_at >= self.seconds
        )

    def seconds_remaining(self, now: float) -> float:
        if self._last_declined_at is None:
            return 0.0
        return max(0.0, self.seconds - (now - self._last_declined_at))

    def mark_staying_idle(self, now: float) -> None:
        self._last_declined_at = now

    def clear(self) -> None:
        self._last_declined_at = None


def _distribution(observations: Iterable[EmotionObservation]) -> dict[str, int]:
    counter = Counter(observation.label for observation in observations)
    return dict(counter.most_common())


def _dominant_emotion(observations: list[EmotionObservation]) -> str:
    if not observations:
        return "neutral"
    counts = Counter(observation.label for observation in observations)
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}
    if len(tied) == 1:
        return next(iter(tied))
    average_by_label = {
        label: sum(
            observation.confidence
            for observation in observations
            if observation.label == label
        )
        / counts[label]
        for label in tied
    }
    return max(average_by_label, key=average_by_label.get)
