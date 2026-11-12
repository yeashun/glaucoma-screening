"""Replay a fixed two-model forward, without caching images or predictions."""
import torch


class GraphForward:
    def __init__(self, models, age):
        self.models,self.age=models,age
        self.input=torch.zeros((1,3,384,384),device=age.device)
        stream=torch.cuda.Stream(device=age.device)
        stream.wait_stream(torch.cuda.current_stream(age.device))
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):self.eager(self.input)
        torch.cuda.current_stream(age.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph),torch.inference_mode():
            self.output=self.eager(self.input)

    def eager(self,image):
        with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
            outputs=[model(image,self.age) for model in self.models]
        return torch.cat(outputs).float().sigmoid()

    @torch.inference_mode()
    def __call__(self,image):
        self.input.copy_(image)
        self.graph.replay()
        return self.output
