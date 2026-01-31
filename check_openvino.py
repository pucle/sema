import onnxruntime as ort
print("Providers available:", ort.get_available_providers())
try:
    sess = ort.InferenceSession("d:/sema/model.onnx", providers=["OpenVINOExecutionProvider"]) # Assuming no model file, this might fail, but checking init
    print("Session created with OpenVINO")
except Exception as e:
    print(f"Test Init Error (Expected if no model): {e}")
