# jetson_server.py
import asyncio
import websockets
import json
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from datetime import datetime

class JetsonDetectionServer:
    def __init__(self):
        self.model = YOLO('best.pt')  # Load your custom YOLO model
        camera_id = "/dev/video1"
        self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)  # Use CSI camera or USB camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.is_detecting = False
        
    async def process_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        # Run YOLO detection
        results = self.model(frame)
        detections = []
        
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result
            detection = {
                "id": str(datetime.now().timestamp()),
                "label": self.model.names[int(cls)],
                "confidence": float(conf),
                "bbox": {
                    "x": float(x1 / frame.shape[1] * 100),
                    "y": float(y1 / frame.shape[0] * 100),
                    "width": float((x2 - x1) / frame.shape[1] * 100),
                    "height": float((y2 - y1) / frame.shape[0] * 100)
                }
            }
            detections.append(detection)
            
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "frame": frame_base64,
            "detections": detections,
            "fps": self.camera.get(cv2.CAP_PROP_FPS)
        }

    async def websocket_handler(self, websocket, path):
        try:
            while True:
                message = await websocket.recv()
                command = json.loads(message)
                
                if command["type"] == "toggle_detection":
                    self.is_detecting = command["value"]
                    await websocket.send(json.dumps({
                        "type": "status",
                        "detecting": self.is_detecting
                    }))
                
                if self.is_detecting:
                    result = await self.process_frame()
                    if result:
                        await websocket.send(json.dumps({
                            "type": "frame",
                            **result
                        }))
                
                await asyncio.sleep(0.033)  # ~30 FPS
                
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            if self.is_detecting:
                self.is_detecting = False

    def start_server(self):
        server = websockets.serve(
            self.websocket_handler,
            "0.0.0.0",  # Listen on all interfaces
            8765  # Port number
        )
        print("WebSocket server started on ws://0.0.0.0:8765")
        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    server = JetsonDetectionServer()
    server.start_server()

