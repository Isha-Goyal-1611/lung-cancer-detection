import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import uuid
import json
from datetime import datetime
from full_pipeline import run_full_pipeline

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Lung Cancer Detection API",
    description="AI-powered lung nodule candidate detection from CT scans",
    version="1.0.0"
)

# In-memory storage for results (in production: use a real database)
results_store = {}

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "message": "Lung Cancer Detection API is running"
    }

@app.post("/analyze")
async def analyze_ct_scan(file: UploadFile = File(...)):
    """
    Upload a DICOM file and get nodule candidates back
    Returns: scan_id, candidates list, warnings
    """
    # Step 1: Validate file type
    if not file.filename.endswith('.dcm'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a DICOM (.dcm) file."
        )
    
    # Step 2: Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    # Step 3: Run pipeline
    try:
        candidates_df, warnings = run_full_pipeline(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(
            status_code=422,
            detail=f"Pipeline failed: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Step 4: Generate unique scan ID
    scan_id = str(uuid.uuid4())[:8]
    
    # Step 5: Prepare results
    results = {
        "scan_id": scan_id,
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "candidates_found": len(candidates_df),
        "warnings": warnings.warnings,
        "candidates": candidates_df.to_dict(orient='records')
            if not candidates_df.empty else []
    }
    
    # Step 6: Store results for later retrieval
    results_store[scan_id] = results
    
    return JSONResponse(content=results)

@app.get("/results/{scan_id}")
def get_results(scan_id: str):
    """
    Retrieve results of a previously analyzed scan
    """
    if scan_id not in results_store:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for scan_id: {scan_id}"
        )
    return JSONResponse(content=results_store[scan_id])

@app.get("/results")
def list_all_results():
    """
    List all analyzed scans
    """
    return {
        "total_scans": len(results_store),
        "scan_ids": list(results_store.keys()),
        "scans": [
            {
                "scan_id": k,
                "timestamp": v["timestamp"],
                "filename": v["filename"],
                "candidates_found": v["candidates_found"]
            }
            for k, v in results_store.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)