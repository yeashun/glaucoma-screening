import hashlib
import json
from pathlib import Path
from torch.hub import download_url_to_file
from torchvision.models import EfficientNet_V2_S_Weights,ConvNeXt_Tiny_Weights,EfficientNet_B0_Weights

root=Path('pretrained')
directory=root/'hub/checkpoints'
directory.mkdir(parents=True,exist_ok=True)
records=[]
for weights in (EfficientNet_V2_S_Weights.IMAGENET1K_V1,ConvNeXt_Tiny_Weights.IMAGENET1K_V1,EfficientNet_B0_Weights.IMAGENET1K_V1):
    url=weights.url
    path=directory/url.rsplit('/',1)[-1]
    expected=path.stem.rsplit('-',1)[-1]
    if not path.exists():
        print(json.dumps({'download':url}),flush=True)
        download_url_to_file(url,str(path)+'.partial',hash_prefix=expected,progress=False)
        Path(str(path)+'.partial').replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(expected):
        raise ValueError(f'Hash mismatch: {path}')
    records.append({'enum':str(weights),'url':url,'sha256':digest,'bytes':path.stat().st_size,
                    'source_data':'ImageNet-1K','license_review':'See official torchvision and original model license; retain full texts.'})
    (root/'weights_manifest.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[-1]),flush=True)
