import torch
import torch.fx
import torch.nn.functional as F
from typing import Callable
import torchdynamo
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import default_qconfig

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale

global prepare_mode
prepare_mode = True
class QuantizationWrapper:
    def __init__(self, model, example):
        super().__init__()
        model.eval()
        self.prepared_model = prepare_fx(model, {"": default_qconfig})
        self.quantized_model = None

    def __call__(self, *args, **kwargs):
        global prepare_mode
        print("prepare mode in QuantizationWrapper:", prepare_mode)
        if prepare_mode:
            print("running prepared model")
            return self.prepared_model(*args, **kwargs)
        else:
            if self.quantized_model is None:
                self.quantized_model = convert_fx(self.prepared_model)
                print("quantized model:", self.quantized_model)
            print("running quantized model")
            return self.quantized_model(*args, **kwargs)                
            
def quant_compiler(model, example):
    return QuantizationWrapper(model, example)
    
def quantize(m, example_inputs):
    torchdynamo.config.debug = True
    global prepare_mode
    prepare_mode = True
    with torchdynamo.optimize(quant_compiler):
        # any PyTorch code
        # fx_prepare() is called to optimize extracted fragments
        # should reach a fixed point where nothing new is compiled
        m(*example_inputs)

    # calibration
    # Optionally:
    with torchdynamo.run():
        # any PyTorch code
        # previosly compiled artifacts are reused
        # provides a quiescence guarantee, without compiles
        m(*example_inputs)

    prepare_mode = False
    with torchdynamo.run():
        # any PyTorch code
        # previosly compiled artifacts are reused
        # provides a quiescence guarantee, without compiles
        m(*example_inputs)

    print(m)

m = BasicModule().eval()
example_inputs = (torch.randn(1, 10),)    
quantize(m, example_inputs)
