# BiSeNet-MegEngine

The MegEngine implementation of BiSeNet.

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't want to compare the ouput error between the MegEngine implementation and PyTorch one, just ignore requirements.txt and install MegEngine from the command line:

```bash
python3 -m pip install --upgrade pip 
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```

### Convert weights

Convert trained weights from torch to megengine, the converted weights will be saved in ./pretained/ , you need to specify the convert model architecture and path to checkpoint offered by [official repo](https://github.com/AlfredXiangWu/LightCNN#evaluation).

```bash
python convert_weights.py -m cityscapes-bisenetv1 -c /path/to/ckpt --num-classes 19
```

### Compare

Use `python compare.py` .

By default, the compare script will convert the torch state_dict to the format that megengine need.

If you want to compare the error by checkpoints, you neet load them manually.

### Load From Hub

Import from megengine.hub:

Way 1:

```python
from megengine import hub

modelhub = hub.import_module(
    repo_info='asthestarsfalll/BiSeNet-MegEngine:main', git_host='github.com')

# custom you own BiSeNet
bisenet = modelhub.BiSeNetV1(n_classes=150)

# load pretrained model
pretrained_model = modelhub.bisenetv1(pretrained=True)
```

Way 2:

```python
from  megengine import hub

# load pretrained model 
model_name = 'bisenetv1'
pretrained_model = hub.load(
    repo_info='asthestarsfalll/BiSeNet-MegEngine:main', entry=model_name, git_host='github.com', pretrained=True)
```

For those models which do not have pretrained model online, you need to convert weights mannaly, and load the model without pretrained weights like this:

```python
model = modelhub.bisenetv1()
# or
model_name = 'bisenetv1'
model = hub.load(
    repo_info='asthestarsfalll/BiSeNet-MegEngine:main', entry=model_name, git_host='github.com')
```

## TODO

- [ ] add BiSeNet v2

## Reference

[The official(maybe) pytorch implementation of BiSeNet](https://github.com/CoinCheung/BiSeNet)