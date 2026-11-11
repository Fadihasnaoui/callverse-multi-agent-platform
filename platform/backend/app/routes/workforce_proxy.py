"""Same-origin access to the separate Workforce API through the shared session."""
import requests
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from app import config

router=APIRouter(prefix='/workforce-api',tags=['workforce'])


@router.api_route('/{path:path}',methods=['GET','POST'])
async def proxy(path: str,request: Request):
    if '..' in path.split('/') or '.' in path.split('/') or not (path=='health' or path.startswith(('biz/','live/','external/'))):
        raise HTTPException(404,'Unknown workforce endpoint')
    body=await request.body()
    try:
        response=await run_in_threadpool(requests.request,request.method,
            f'{config.WORKFORCE_API_URL}/{path}',params=list(request.query_params.multi_items()),
            data=body or None,headers={'Content-Type':'application/json'},timeout=65,allow_redirects=False)
    except requests.Timeout:
        raise HTTPException(504,'Workforce response timed out. Refresh its state before repeating an action.')
    except requests.RequestException:
        raise HTTPException(502,'Workforce API is unavailable. Check that its service is running.')
    return Response(content=response.content,status_code=response.status_code,
                    media_type=response.headers.get('Content-Type','application/json'))
