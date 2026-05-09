
import streamlit as st
import requests

#title
st.title("Pneumonia Detection WebApp")

st.write("Upload clinical Excel data and X-ray image to detect pneumonia.")

#upload files
clinical_file = st.file_uploader("Upload Clinical Excel File",type=["xlsx"])
xray_file = st.file_uploader("Upload X-ray Image",type=["jpg","jpeg","png"])

#submit
if st.button("Analyze"):

    if clinical_file is None or xray_file is None:
        st.warning("Please upload both files!")
    else:

        files = {
            "clinical_file":(clinical_file.name,clinical_file,"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            "xray_image":(xray_file.name,xray_file,"image/jpeg")
        }

        try:
            #call fastapi bcknd
            response = requests.post("http://127.0.0.1:8000/analyze/",files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("Analysis Complete!")

                #display final diagnosis
                st.subheader("Final Diagnosis")
                st.write(result.get("final_diagnosis"))

                #Display agent sumary
                st.subheader("Clinical Data Analysis")
                st.json(result.get("clinical_summary"))

                st.subheader("X-ray Analysis")
                st.json(result.get("xray_summary"))

            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
