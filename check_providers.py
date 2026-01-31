import onnxruntime as ort
print("--- Providers ---")
for p in ort.get_available_providers():
    print(p)
print("--- End Providers ---")
