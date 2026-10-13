# VHDiff

**Comorbidity Controlled Heart Sound Synthesis for Coexisting Valvular Heart Diseases**

VHDiff generates heart sound waveforms for individual valvular diseases and their combinations. Choose any complete configuration of five disease attributes and generate three second, mono WAV clips at 2 kHz.

**[Listen to the audio gallery](https://zhj12399.github.io/VHDiff/)** · **[Explore real database examples](https://zhj12399.github.io/VHDiff/real.html)**

This inference release contains the model weights and the code required to generate audio. The gallery shows two generated examples for each of the 32 configurations, with waveform, Log-Mel and MFCC views. It identifies 20 configurations observed in the clinical database and 12 absent configurations. A separate page presents 40 real database examples.

## Installation

Python 3.11 and a CUDA capable GPU are recommended. CPU generation is also supported.

```bash
git clone https://github.com/zhj12399/VHDiff.git
cd VHDiff
pip install -r requirements.txt
```

For a CUDA installation appropriate to your GPU, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Model weights

The inference checkpoint is included at [`checkpoints/vhdiff.pt`](checkpoints/vhdiff.pt). It contains the final generator parameters and the configuration needed to reconstruct the model.

- SHA256: `8ea89552e3882ad58e9366bd4e1ae48250826a340e323188a0278322d3f445ef`
- Size: approximately 99 MiB
- Output: 2,000 Hz, 3 seconds, 6,000 waveform samples

## Generate audio

Generate two clips for a single disease:

```bash
python generate.py --condition MR --num-samples 2 --output outputs/mr
```

Generate a selected comorbidity:

```bash
python generate.py --condition AS+MS --num-samples 2 --output outputs/as_ms
```

Generate normal requests and several disease combinations in one run:

```bash
python generate.py --condition Normal --condition AS,MR,TR --condition MS,MR --num-samples 2 --output outputs/examples
```

Generate every configuration:

```bash
python generate.py --all --num-samples 2 --output outputs/all
```

Each run saves WAV files in folders named for their requested diseases. `generation.json` records the requests and sampling settings. Fresh independent random noise is used by default. An optional `--seed` enables replay, and `--device cpu` selects CPU inference.

### Disease attributes

| Attribute | Disease |
|---|---|
| AS | Aortic stenosis |
| AR | Aortic regurgitation |
| MS | Mitral stenosis |
| MR | Mitral regurgitation |
| TR | Tricuspid regurgitation |

Conditions can also be specified as five bit codes in the order **AS, AR, MS, MR, TR**. For example, `10100` requests AS + MS and `00000` requests Normal. Every specified code defines all five attributes, including their absence.

### Batch generation

```bash
python generate.py --condition AS+MS --num-samples 4 --batch-size 4 --device cuda
```

The CLI exports the native 2 kHz waveform.

## Repository layout

```text
VHDiff/
├── checkpoints/vhdiff.pt       Model weights
├── generate.py                Generation CLI
├── vhdiff/                    Architecture and inference sampler
├── requirements.txt           Inference dependencies
└── docs/                      Audio gallery
```

## Acknowledgements

We thank Beijing Anzhen Hospital for its support. We also thank the teams behind [PhysioNet/CinC 2016](https://physionet.org/content/challenge-2016/1.0.0/) and [CirCor DigiScope](https://physionet.org/content/circor-heart-sound/1.0.3/).
