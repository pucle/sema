import onnxruntime as ort
for p in ort.get_available_providers():
    print(p)
