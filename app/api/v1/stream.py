from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from app.services.stream_service import StreamService

router = APIRouter(prefix="/anpr", tags=["Stream"])

@router.get("/stream")
async def stream_video(url: str = Query(..., description="RTSP URL or '0' for webcam")):
    return StreamingResponse(
        StreamService.get_video_stream(url),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )