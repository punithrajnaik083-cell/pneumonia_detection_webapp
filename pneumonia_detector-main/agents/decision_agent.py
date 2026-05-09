
def make_final_decision(clinical_result: dict, xray_result: dict) -> dict:
   

    if clinical_result["pneumonia_risk"]>0.5 and xray_result["prediction"]=="Pneumonia":
        diagnosis =  "Pneumonia Detected"
    else:
        diagnosis = "No Pneumonia Detected"

    final_result={
        "final_diagnosis":diagnosis,
        "clinical_summary":clinical_result,
        "xray_summary":xray_result
    }

    return final_result