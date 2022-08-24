import time

import megengine as mge
import numpy as np
import torch

from models.bisenetv1 import bisenetv1
from models.torch_models import BiSeNetV1 as torch_BiSeNetV1
from convert_weights import convert


mge_model = bisenetv1(pretrained=True, n_classes=19)
torch_model = torch_BiSeNetV1(n_classes=19)

# torch_model.load_state_dict(torch.load('./model_final_v1_city_new.pth', map_location='cpu'))
# mge_model.load_state_dict(mge.load('./pretrained/cityscapes-bisenetv1.pkl'))

s = torch_model.state_dict()
m = convert(torch_model, s)
mge_model.load_state_dict(m)

mge_model.eval()
torch_model.eval()

torch_time = meg_time = 0.0

if torch.cuda.is_available():
    torch_model.cuda()

def test_func(mge_out, torch_out, post_process=None):
    if torch.cuda.is_available():
        torch_out = torch_out.detach().cpu().numpy()
    else:
        torch_out = torch_out.detach().numpy()
    mge_out = mge_out.numpy()
    if post_process is not None:
        mge_out = post_process(mge_out)
        torch_out = post_process(torch_out)
    result = np.isclose(mge_out, torch_out, rtol=1e-3)
    ratio = np.mean(result)
    allclose = np.all(result) > 0
    abs_err = np.mean(np.abs(mge_out - torch_out))
    std_err = np.std(np.abs(mge_out - torch_out))
    return ratio, allclose, abs_err, std_err


def argmax(logits):
    return np.argmax(logits, axis=1)

for i in range(15):
    results = []
    inp = np.random.randn(2, 3, 224, 224)
    mge_inp = mge.tensor(inp, dtype=np.float32)
    torch_inp = torch.tensor(inp, dtype=torch.float32)

    if torch.cuda.is_available():
        torch_inp = torch_inp.cuda()

    st = time.time()
    mge_out = mge_model(mge_inp)[0] # final output, others for aux training
    meg_time += time.time() - st

    st = time.time()
    torch_out = torch_model(torch_inp)[0]
    torch_time += time.time() - st

    ratio, allclose, abs_err, std_err = test_func(mge_out, torch_out, argmax)
    results.append(allclose)
    print(f"Result: {allclose}, {ratio*100 : .4f}% elements is close enough\n which absolute error is  {abs_err} and absolute std is {std_err}")

assert all(results), "not aligned"

print(f"meg time: {meg_time}, torch time: {torch_time}")
