import os
import cv2
import copy
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile
import uuid

app = FastAPI(title="Face Swap API", description="Face swap service using InsightFace")

def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model

def getFaceAnalyser(model_path: str, providers, det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser

def get_one_face(face_analyser, frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

def get_many_faces(face_analyser, frame: np.ndarray):
    """get faces from left to right by order"""
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None

def swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    """paste source_face on target image"""
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

def process_face_swap(source_imgs: List[Image.Image], target_img: Image.Image, 
                     source_indexes: str, target_indexes: str, model_path: str):
    # load machine default available providers
    providers = onnxruntime.get_available_providers()

    # load face_analyser
    face_analyser = getFaceAnalyser(model_path, providers)
    
    # load face_swapper
    face_swapper = getFaceSwapModel(model_path)
    
    # read target image
    target_img_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # detect faces that will be replaced in the target image
    target_faces = get_many_faces(face_analyser, target_img_cv)
    
    if target_faces is None:
        raise HTTPException(status_code=400, detail="No target faces found!")
        
    num_target_faces = len(target_faces)
    num_source_images = len(source_imgs)

    temp_frame = copy.deepcopy(target_img_cv)
    
    if isinstance(source_imgs, list) and num_source_images == num_target_faces:
        print("Replacing faces in target image from the left to the right by order")
        for i in range(num_target_faces):
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_imgs[i]), cv2.COLOR_RGB2BGR))
            source_index = i
            target_index = i

            if source_faces is None:
                raise HTTPException(status_code=400, detail="No source faces found!")

            temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame)
            
    elif num_source_images == 1:
        # detect source faces that will be replaced into the target image
        source_faces = get_many_faces(face_analyser, cv2.cvtColor(np.array(source_imgs[0]), cv2.COLOR_RGB2BGR))
        
        if source_faces is None:
            raise HTTPException(status_code=400, detail="No source faces found!")
            
        num_source_faces = len(source_faces)

        if target_indexes == "-1":
            if num_source_faces == 1:
                print("Replacing all faces in target image with the same face from the source image")
                num_iterations = num_target_faces
            elif num_source_faces < num_target_faces:
                print("There are less faces in the source image than the target image, replacing as many as we can")
                num_iterations = num_source_faces
            elif num_target_faces < num_source_faces:
                print("There are less faces in the target image than the source image, replacing as many as we can")
                num_iterations = num_target_faces
            else:
                print("Replacing all faces in the target image with the faces from the source image")
                num_iterations = num_target_faces

            for i in range(num_iterations):
                source_index = 0 if num_source_faces == 1 else i
                target_index = i
                temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame)
        else:
            print("Replacing specific face(s) in the target image with specific face(s) from the source image")

            if source_indexes == "-1":
                source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

            if target_indexes == "-1":
                target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

            source_indexes = source_indexes.split(',')
            target_indexes = target_indexes.split(',')
            num_source_faces_to_swap = len(source_indexes)
            num_target_faces_to_swap = len(target_indexes)

            if num_source_faces_to_swap > num_source_faces:
                raise HTTPException(status_code=400, detail="Number of source indexes is greater than the number of faces in the source image")

            if num_target_faces_to_swap > num_target_faces:
                raise HTTPException(status_code=400, detail="Number of target indexes is greater than the number of faces in the target image")

            if num_source_faces_to_swap > num_target_faces_to_swap:
                num_iterations = num_source_faces_to_swap
            else:
                num_iterations = num_target_faces_to_swap

            if num_source_faces_to_swap == num_target_faces_to_swap:
                for index in range(num_iterations):
                    source_index = int(source_indexes[index])
                    target_index = int(target_indexes[index])

                    if source_index > num_source_faces-1:
                        raise HTTPException(status_code=400, detail=f"Source index {source_index} is higher than the number of faces in the source image")

                    if target_index > num_target_faces-1:
                        raise HTTPException(status_code=400, detail=f"Target index {target_index} is higher than the number of faces in the target image")

                    temp_frame = swap_face(face_swapper, source_faces, target_faces, source_index, target_index, temp_frame)
    else:
        raise HTTPException(status_code=400, detail="Unsupported face configuration")
    
    result_image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    return result_image

@app.get("/")
async def root():
    return {"message": "Face Swap API is running!"}

@app.post("/swap_faces")
async def swap_faces(
    target_image: UploadFile = File(..., description="Target image file"),
    source_images: List[UploadFile] = File(..., description="Source image files"),
    source_indexes: str = Form(default="-1", description="Comma separated list of source face indexes"),
    target_indexes: str = Form(default="-1", description="Comma separated list of target face indexes")
):
    """
    Swap faces between source and target images
    """
    try:
        # Check if model exists
        model_path = "./checkpoints/inswapper_128.onnx"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Model file not found. Please ensure inswapper_128.onnx is in ./checkpoints/")

        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        
        if target_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Target image must be JPEG or PNG")
        
        for source_img in source_images:
            if source_img.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail="Source images must be JPEG or PNG")

        # Load images
        target_img_data = await target_image.read()
        target_img = Image.open(tempfile.BytesIO(target_img_data))
        
        source_imgs = []
        for source_img in source_images:
            source_img_data = await source_img.read()
            source_imgs.append(Image.open(tempfile.BytesIO(source_img_data)))

        # Process face swap
        result_image = process_face_swap(source_imgs, target_img, source_indexes, target_indexes, model_path)
        
        # Save result to temporary file
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        result_image.save(temp_output.name)
        temp_output.close()
        
        return FileResponse(
            temp_output.name,
            media_type="image/png",
            filename=f"face_swap_result_{uuid.uuid4().hex[:8]}.png"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face swap processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_path = "./checkpoints/inswapper_128.onnx"
    model_exists = os.path.exists(model_path)
    
    return {
        "status": "healthy" if model_exists else "unhealthy",
        "model_loaded": model_exists,
        "providers": onnxruntime.get_available_providers()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)