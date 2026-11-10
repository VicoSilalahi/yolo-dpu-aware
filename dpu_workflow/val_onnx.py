import onnxruntime

from ultralytics.models.yolo.segment import SegmentationValidator


class ONNXSegValidator(SegmentationValidator):
    def __init__(self, onnx_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def init_metrics(self, model):
        super().init_metrics(model)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        return batch


if __name__ == "__main__":
    pass
