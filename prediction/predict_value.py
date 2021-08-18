class Prediction:

    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline

    def predict(self, data):
        data = self.pipeline.transform(data)
        pred = self.model.predict(data)
        return pred[0]

