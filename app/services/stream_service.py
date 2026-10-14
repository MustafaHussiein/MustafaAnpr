import time
import cv2
from app.core.logging_config import logger


class StreamService:
    @staticmethod
    def get_video_stream(url: str):
        # Open RTSP link or local webcam (0 if empty/0)
        camera_source = 0 if url == "0" or not url else url
        is_rtsp = isinstance(camera_source, str) and camera_source.startswith("rtsp")

        logger.info(f"Opening video stream source: {camera_source!r}")

        # FFMPEG backend handles RTSP far more reliably than the default,
        # and a small buffer avoids the multi-second lag/freeze RTSP streams
        # are notorious for.
        cap = (
            cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
            if is_rtsp
            else cv2.VideoCapture(camera_source)
        )
        if is_rtsp:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.error(f"Could not open stream: {camera_source!r}")
            cap.release()
            return

        consecutive_failures = 0
        max_consecutive_failures = 30  # ~ a few seconds of dropped frames

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    consecutive_failures += 1
                    logger.warning(
                        f"Failed to read frame from {camera_source!r} "
                        f"({consecutive_failures}/{max_consecutive_failures})"
                    )
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            f"Too many consecutive read failures on "
                            f"{camera_source!r}, closing stream."
                        )
                        break
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception:
            logger.exception(f"Error while streaming from {camera_source!r}")
        finally:
            cap.release()
            logger.info(f"Closed video stream source: {camera_source!r}")
