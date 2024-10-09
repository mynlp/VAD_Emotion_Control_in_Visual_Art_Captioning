# VAD Emotion Control in Visual Art Captioning via Disentangled Multi-modal Representation

This is the source code of our paper:
```
Ryo Ueda, Hiromi Narimatsu, Yusuke Miyao, & Shiro Kumano.
VAD Emotion Control in Visual Art Captioning via Disentangled Multimodal Representation.
ACII2024.
```

## Requirements
- Install `rye` to install and manage Python packages from https://github.com/astral-sh/rye and run `rye sync` to install all necessary packages
- Download `ArtEmis` dataset from https://www.artemisdataset.org/
- Download `Wikiart` dataset from, e.g., https://github.com/cs-chan/ArtGAN
- Download `NRC-VAD` dataset from https://saifmohammad.com/WebPages/nrc-vad.html or by git cloning https://github.com/SteveKGYang/VAD-VAE.git
- Git clone `D-ViSA` dataset from https://github.com/dxlabskku/D-ViSA.git

## Train model
Minial example command:
```
$ .venv/bin/python -m src.train.train \
    --artemis_data_path ${Path to ArtEmis} \
    --artemis_data_sep ${Appropriate Delimiter} \
    --nrc_vad_lexicon_data_path ${Path to NRC-VAD} \
    --dvisa_data_path ${Path to D-ViSA} \
    --wikiart_dirpath ${Path to WikiArt}
```

For more options, try:
```
$ .venv/bin/python -m src.train.train --help
```
or directly check out `./src/train/train.py`.

## Citation

```
@inproceedings{UedaNMK2024,
  author={Ueda, Ryo and Narimatsu, Hiromi and Miyao, Yusuke and Kumano, Shiro},
  booktitle={2024 12th International Conference on Affective Computing and Intelligent Interaction (ACII)}, 
  title={VAD Emotion Control in Visual Art Captioning via Disentangled Multimodal Representation},
  year={2024}
}
```
