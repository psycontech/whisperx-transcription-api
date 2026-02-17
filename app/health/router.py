from app.common.router import VersionRouter
from app.common.response import HttpResponse

router = VersionRouter(version="1", path="health", tags=["Health"])

@router.get("/")
async def health_check() -> HttpResponse[None]:
    return HttpResponse(message="Health check", data=None)
