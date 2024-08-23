import uvicorn
from fastapi import FastAPI, HTTPException, status, Body
from pydantic import BaseModel

import asyncio
from test2 import audio_detect

tasks={}

app = FastAPI()


class AudioStart(BaseModel):
    SN:str
    ip:str



@app.get("/")
async def home():
    return {
        "Message" : "hi"
    }

@app.post("/start")
async def audio_start(req:AudioStart):  
    if req.SN in tasks:
        task=tasks[req.SN]
        if not task.done():
            raise HTTPException(
                status_code=status.HTTP_400_NOT_FOUND,
                detail="already detecting"
            )
        del tasks[req.SN]
    tasks[req.SN]=asyncio.create_task(audio_detect(req.ip))       ####  detecting func

    return {
        "message" : "suc"
    }

@app.post("/terminate")
async def vision_terminate(SN:str= Body(..., embed=True)):
    task=tasks[SN]
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="task not found"
        )
    elif task.done():
        del tasks[SN]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="already detecting"
        )
    tasks[SN].cancel()
    return {
        "message" : "suc"
    }



if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)