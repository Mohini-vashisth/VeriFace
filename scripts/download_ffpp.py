import kagglehub, os, sys, shutil

print("⏳ Starting kagglehub download for xdxd003/ff-c23 ...")
try:
    path = kagglehub.dataset_download("xdxd003/ff-c23")
except Exception as e:
    print("❌ kagglehub download failed with:", repr(e))
    sys.exit(2)

print("✅ kagglehub finished.")
print("Path to dataset files:", path)
# list top-level contents (safe, limited output)
try:
    items = os.listdir(path)
    print("Top-level contents:", items[:40])
except Exception as e:
    print("Could not list contents:", e)
# also print full absolute path for convenience
print("Absolute path:", os.path.abspath(path))
