ONNX parser: Takes a converted PyTorch trained model into the ONNX format as input and populates a network object in TensorRT. 
Builder: Takes a network in TensorRT and generates an engine that is optimized for the target platform. 
Engine: Takes input data, performs inferences, and emits inference output.
Logger: Associated with the builder and engine to capture errors, warnings, and other information during the build and inference phases.
