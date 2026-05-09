
from fastapi import FastAPI , UploadFile, File
from fastapi.responses import JSONResponse
from agents.clinical_agent import analyze_clinical_data
from agents.xray_agent import analyze_xray_image
from agents.decision_agent import make_final_decision

app = FastAPI(title="Pneumonia Detection API")

@app.get("/")
def read_root():
    return {"message": "Pneumonia detection API is running!"}


@app.post("/analyze/")
async def analyze(clinical_file: UploadFile = File(...),
                  xray_image: UploadFile = File(...)):
    

    #save uploaded files temporarily
    clinical_path = f"temp_{clinical_file.filename}"
    xray_path = f"temp_{xray_image.filename}"

    with open(clinical_path,"wb") as f:
        f.write(await clinical_file.read())

    with open(xray_path,"wb") as f:
        f.write(await xray_image.read())

    #ryn agemnts
    clinical_result = analyze_clinical_data(clinical_path)
    xray_result = analyze_xray_image(xray_path)

    #final decision
    final_result = make_final_decision(clinical_result,xray_result)

    #return json response
    return JSONResponse(content=final_result)