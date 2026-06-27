import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydicom
import pandas as pd
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from full_pipeline import run_full_pipeline

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Detection AI",
    page_icon="🫁",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────
st.title("🫁 Lung Cancer Detection System")
st.markdown("*AI-powered nodule candidate detection from CT scans*")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Higher threshold = fewer but more confident detections"
    )
    st.divider()
    st.markdown("**About this tool:**")
    st.markdown("This AI tool assists radiologists in detecting suspicious nodule candidates in chest CT scans.")
    st.warning("⚠️ For research use only. Not for clinical diagnosis.")

# ── File Upload ──────────────────────────────────────────────
st.header("📁 Upload CT Scan")
uploaded_file = st.file_uploader(
    "Upload a DICOM file (.dcm)",
    type=['dcm'],
    help="Upload a chest CT scan in DICOM format"
)

if uploaded_file is not None:
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Run pipeline with caching
    with st.spinner("🔄 Processing CT scan... please wait"):
        try:
            candidates_df, warnings = run_full_pipeline(tmp_path)
            st.success("✅ Pipeline complete!")
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            st.stop()
    
    # Show warnings if any
    if warnings.has_warnings():
        for w in warnings.warnings:
            st.warning(f"⚠️ {w}")
    
    # Load DICOM for display
    ds = pydicom.dcmread(tmp_path, force=True)
    pixel_array = ds.pixel_array
    slope = float(ds.RescaleSlope)
    intercept = float(ds.RescaleIntercept)
    hu_image = np.clip(pixel_array * slope + intercept, -1000, 400)
    
    st.divider()
    
    # ── Results Layout ────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Candidates Found", len(candidates_df))
    with col2:
        st.metric("Warnings", len(warnings.warnings))
    with col3:
        st.metric("Scan Shape", f"{pixel_array.shape[0]}×{pixel_array.shape[1]}")
    
    st.divider()
    
    # ── CT Scan Display ───────────────────────────────────────
    st.header("🔬 CT Scan Analysis")
    
    img_col, report_col = st.columns([2, 1])
    
    with img_col:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original CT
        axes[0].imshow(hu_image, cmap='gray')
        axes[0].set_title('CT Scan (HU)')
        axes[0].axis('off')
        
        # CT with candidates marked
        axes[1].imshow(hu_image, cmap='gray')
        if not candidates_df.empty:
            axes[1].scatter(
                candidates_df['x'],
                candidates_df['y'],
                c='red', s=100, marker='x',
                linewidths=2, label='Candidates'
            )
            # Draw circles around candidates
            for _, row in candidates_df.iterrows():
                circle = plt.Circle(
                    (row['x'], row['y']),
                    radius=15,
                    color='red',
                    fill=False,
                    linewidth=2
                )
                axes[1].add_patch(circle)
        
        axes[1].set_title(f'Detected Candidates ({len(candidates_df)})')
        axes[1].axis('off')
        if not candidates_df.empty:
            axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with report_col:
        st.subheader("📋 Candidates Report")
        
        if candidates_df.empty:
            st.info("No candidates found above threshold.")
        else:
            # Display candidates table
            display_df = candidates_df.copy()
            display_df['x'] = display_df['x'].round(1)
            display_df['y'] = display_df['y'].round(1)
            display_df['area'] = display_df['area'].astype(int)
            display_df['mean_intensity'] = display_df['mean_intensity'].round(1)
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = candidates_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name="nodule_candidates.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    # ── Patient Info ──────────────────────────────────────────
    st.header("👤 Patient Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Patient ID",
                 str(ds.get('PatientID', 'N/A')))
    with info_col2:
        st.metric("Patient Age",
                 str(ds.get('PatientAge', 'N/A')))
    with info_col3:
        st.metric("Scanner",
                 str(ds.get('Manufacturer', 'N/A')))
    
    # Cleanup temp file
    os.unlink(tmp_path)

else:
    # Show instructions when no file uploaded
    st.info("👆 Upload a DICOM (.dcm) file to begin analysis")
    
    st.markdown("""
    ### How to use this tool:
    1. **Upload** a chest CT scan in DICOM format
    2. **Wait** for the AI pipeline to process the scan
    3. **Review** the detected nodule candidates
    4. **Download** the CSV report for your records
    
    ### What this tool detects:
    - Suspicious nodule candidates inside lung tissue
    - Candidates are marked with red circles on the CT image
    - Each candidate includes position, size, and density information
    """)