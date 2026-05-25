import time

from companion_ai.config import load_settings
from companion_ai.conversation import CompanionConversation
from companion_ai.emotion.detector import WebcamEmotionDetector
from companion_ai.emotion.signal import (
    EmotionSignalSettings,
    EmotionSignalSummary,
    IdleDecisionCooldown,
    RollingEmotionSignal,
)
from companion_ai.providers.llm import GroqChatProvider
from companion_ai.providers.stt import GroqSpeechToTextProvider
from companion_ai.providers.tts import LocalTextToSpeechProvider
from companion_ai.tools.registry import default_registry


def _format_signal_log(summary: EmotionSignalSummary) -> str:
    distribution = summary.distribution_text or "none"
    return (
        "Emotion signal: "
        f"{summary.non_neutral_frames}/{summary.total_frames} non-neutral, "
        f"avg={summary.average_confidence:.2f}, "
        f"max={summary.max_confidence:.2f}, "
        f"dominant={summary.dominant_emotion}, "
        f"distribution={distribution}"
    )


def main() -> None:
    settings = load_settings()

    emotion_detector = WebcamEmotionDetector(
        camera_index=settings.camera_index,
        emotion_model_path=settings.emotion_model_path,
    )
    conversation = CompanionConversation(
        llm=GroqChatProvider(settings.groq_api_key, settings.groq_chat_model),
        stt=GroqSpeechToTextProvider(settings.groq_api_key, settings.groq_stt_model),
        tts=LocalTextToSpeechProvider(),
        tools=default_registry(),
        voice_sample_seconds=settings.voice_sample_seconds,
        silence_threshold=settings.silence_threshold,
        silence_seconds=settings.silence_seconds,
        max_listen_seconds=settings.max_listen_seconds,
    )

    emotion_signal = RollingEmotionSignal(
        EmotionSignalSettings(
            window=settings.emotion_signal_window,
            min_non_neutral=settings.emotion_signal_min_non_neutral,
            min_avg_confidence=settings.emotion_signal_min_avg_confidence,
            spike_confidence=settings.emotion_signal_spike_confidence,
            spike_frames=settings.emotion_signal_spike_frames,
        )
    )
    idle_cooldown = IdleDecisionCooldown(settings.idle_decision_cooldown_seconds)
    logged_threshold_signal = False
    logged_cooldown_block = False

    print("Companion AI running. Press q in the video window or Ctrl+C to stop.")
    try:
        if not emotion_detector.webcam_available:
            conversation.tts.speak(
                "I can't access a webcam right now, so I'll run in chat-only mode. "
                "Tell me what's on your mind."
            )
            while True:
                conversation.listen_and_respond()

        if not emotion_detector.available:
            conversation.tts.speak(
                "I can access the webcam, but the emotion model is not loaded, "
                "so emotion triggers are off. Tell me what's on your mind."
            )
            while True:
                conversation.listen_and_respond()

        while True:
            observation = emotion_detector.read_emotion(show_frame=True)
            if emotion_detector.stop_requested:
                break

            if conversation.is_active():
                if observation:
                    conversation.listen_and_respond(
                        emotion_hint=observation.label,
                        confidence=observation.confidence,
                    )
                else:
                    conversation.listen_and_respond()
                emotion_signal.clear()
                idle_cooldown.clear()
                logged_threshold_signal = False
                logged_cooldown_block = False
                continue

            if observation:
                print(f"Emotion: {observation.label} ({observation.confidence:.2f})")
                summary = emotion_signal.add(observation)

                if summary.threshold_met and not logged_threshold_signal:
                    print(_format_signal_log(summary))
                    logged_threshold_signal = True

                now = time.monotonic()
                if summary.threshold_met and idle_cooldown.can_consider(now):
                    print("Emotion signal threshold reached: asking AI to decide.")
                    started = conversation.consider_emotion_observation(
                        emotion=summary.dominant_emotion,
                        average_confidence=summary.average_confidence,
                        max_confidence=summary.max_confidence,
                        non_neutral_frames=summary.non_neutral_frames,
                        total_frames=summary.total_frames,
                        distribution=summary.distribution_text,
                    )
                    emotion_signal.clear()
                    logged_threshold_signal = False
                    logged_cooldown_block = False
                    if started:
                        idle_cooldown.clear()
                        conversation.listen_and_respond(
                            emotion_hint=summary.dominant_emotion,
                            confidence=summary.average_confidence,
                        )
                    else:
                        idle_cooldown.mark_staying_idle(now)
                        print("Consideration result: staying idle; cooldown started.")
                elif summary.threshold_met and not logged_cooldown_block:
                    seconds_remaining = idle_cooldown.seconds_remaining(now)
                    print(
                        "Emotion signal cooldown active: "
                        f"{max(0.0, seconds_remaining):.1f}s remaining."
                    )
                    logged_cooldown_block = True
                elif not summary.threshold_met:
                    logged_threshold_signal = False
                    logged_cooldown_block = False

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("Stopping Companion AI...")
    finally:
        emotion_detector.release()


if __name__ == "__main__":
    main()
