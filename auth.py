# auth.py
import ee

# Initialize with your Google Cloud Project ID
try:
    ee.Initialize(project='earth-engine-project')
    print("✅ Earth Engine initialized with project")
except Exception as e:
    print("❌ Failed to initialize with project:", str(e))
    print("👉 Run ee.Authenticate() again if needed")