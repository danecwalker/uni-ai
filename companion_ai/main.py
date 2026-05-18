import time

from companion_ai.config import load_settings
from companion_ai.conversation import CompanionConversation
from companion_ai.emotion.detector import WebcamEmotionDetector
from companion_ai.providers.llm import GroqChatProvider
from companion_ai.providers.stt import GroqSpeechToTextProvider
from companion_ai.providers.tts import LocalTextToSpeechProvider
from companion_ai.tools.registry import default_registry


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

    matching_frames = 0
    has_started_check_in = False

    print("Companion AI running. Press Ctrl+C to stop.")
    try:
        if not emotion_detector.available:
            conversation.tts.speak(
                "I can't access a webcam right now, so I'll run in chat-only mode. "
                "Tell me what's on your mind."
            )
            while True:
                conversation.listen_and_respond()

        while True:
            observation = emotion_detector.read_emotion()
            if observation:
                print(f"Emotion: {observation.label} ({observation.confidence:.2f})")
                if (
                    observation.label == settings.emotion_trigger
                    and observation.confidence >= settings.emotion_confidence
                ):
                    matching_frames += 1
                else:
                    matching_frames = 0

                if not has_started_check_in and matching_frames >= settings.emotion_frames_required:
                    has_started_check_in = True
                    conversation.start_emotion_check_in(
                        observation.label,
                        observation.confidence,
                    )

            time.sleep(0.25)
    except KeyboardInterrupt:
        print("Stopping Companion AI...")
    finally:
        emotion_detector.release()


if __name__ == "__main__":
    main()
