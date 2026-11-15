# tinygroove

A drum machine with no samples inside. Every kick, clap and hat is made from scratch in your
browser by a 5-million-parameter neural network, in about a second for a full pack of sixteen.

![The TG-16 sampler: a silver box with sixteen pads, a lime LCD and a small creature living in the screen](docs/tg16.jpg)

The model is a student. A large text-to-audio model (Stable Audio 3) played teacher: it made
about 50,000 one-second house drum hits, and a model small enough to ship in a web page learned
to make sounds like them.

## Try it

Play it at **[tinygroove.com](https://tinygroove.com)**, or run it locally:

```bash
python3 demo/serve.py
```

Then open http://localhost:8765. Pads play on click or on `1234 / qwer / asdf / zxcv`; space
starts the groove. It needs a recent Chrome, Edge or other WebGPU browser to be fast, and falls
back to the CPU elsewhere. Nothing leaves your machine.

## How a drum is made

```
noise  ──▶  generator  ──▶  sound code  ──▶  decoder  ──▶  drum
512 numbers   16 steps     32 × 16 numbers    1 pass      1 second
```

Each pad starts as random numbers. The generator nudges them sixteen times toward the request
(drum type, genre, machine, and optionally length, tone and pitch). The decoder turns the result
into audio. Turning a knob keeps a pad's noise and changes only the request, so the sound shifts
but stays the same drum. "New pack" rolls fresh noise for everything.

| | |
|---|---|
| Generator | flow-matching transformer, 4.24M parameters |
| Decoder | spectrogram decoder, 1.02M parameters |
| Download | 22 MB for both |
| Output | 1 s, mono, 44.1 kHz |
| Drum types | 16 |
| Prompt words | 172, including 27 genres and 22 drum machines |
| Training data | 48,741 clips from the teacher |
| Training time | 85 + 19 minutes on one H100 |

## Rebuild it

Each step is one script. You need [uv](https://docs.astral.sh/uv/), a [Modal](https://modal.com)
account for the GPU steps, and a Hugging Face token with access to Stable Audio 3 stored as a
Modal secret named `huggingface`.

| Step | Command | What you get |
|---|---|---|
| 1. Render the teacher's drums | `modal run modal_gen.py --n 50000 --out renders/dataset_v4` | 50,004 raw clips (about 55 GPU-minutes) |
| 2. Gate and clean | `uv run --with numpy python process_dataset.py renders/dataset_v4` | 48,741 clean one-second clips |
| 3. Train the autoencoder | `modal run modal_ae.py --steps 80000 --max-min 85` | `ae_v2/` |
| 4. Train the generator | `modal run modal_flow.py` | `flow_v1/` |
| 5. Export for the browser | `uv run --with torch --with numpy --with onnx --with onnxruntime python export_onnx.py` | `demo/models/` |

The two trained checkpoints (`ae_v2/ae.pt`, `flow_v1/flow.pt`) are in the repository, so step 5
works without the first four.

## What is where

| | |
|---|---|
| `demo/` | the sampler: one HTML file, the exported models, ONNX Runtime Web |
| `wrangler.jsonc`, `.github/workflows/deploy.yml` | deploys `demo/` to tinygroove.com on every push to `main` |
| `gen_batch.py`, `modal_gen.py` | prompt planner and dataset renderer (Mac and Modal) |
| `vocab.py`, `vocab_screen.py`, `vocab_catalogue.csv` | the prompt vocabulary and what each word measurably does |
| `process_dataset.py` | the gate: drops clipped, silent and double-hit clips, trims and normalises |
| `ae_model.py`, `ae_train.py`, `modal_ae.py` | the drum autoencoder |
| `flow_model.py`, `flow_train.py`, `flow_eval.py`, `modal_flow.py` | the generator and its checks |
| `export_onnx.py` | PyTorch to ONNX, verified against PyTorch |
| `extras/` | early helpers: test renders on a Mac, a loop sequencer |

## Known faults

- The cowbells are not good.
- Kicks carry less sub-bass than a real 808.
- Variety within a drum type is about 85% of the teacher's.
- One second of mono is all it knows.

## Thanks

To [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) for teaching,
[tiny-voca](https://github.com/n4ze3m/tiny-voca) for the idea of a very small student, and
[ONNX Runtime Web](https://onnxruntime.ai) for running it in a tab.

## Licence

MIT. See [LICENSE](LICENSE).
