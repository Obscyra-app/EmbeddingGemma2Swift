# EmbeddingGemma 2 — MLX Swift Text + Images + Audio

Native text, image and audio embeddings on Apple Silicon. This package implements Google EmbeddingGemma 2's text encoder, vision tower and audio Conformer, including bidirectional local/global attention, per-layer inputs, 2D vision RoPE, spatial pooling, learned projection and normalized Matryoshka embeddings. No Python is required by the Swift application.

Development status: text, images, audio and the selected 8-bit text/vision profile validated on MacBook Air M3 (16 GiB) and a physical iPhone 16 Pro Max (iOS 27.0). macOS BF16/FP32 passed against the official PyTorch FP32 implementation. Four image fixtures cover budgets 70/140/280, aspect ratios, up/downsampling; pixel preparation, tokens and vectors were checked separately. iPhone image vectors exactly matched Mac BF16 on these inputs. Video and interleaved multimodal documents are outside this version.

## Quick start

- [Swift source and releases](https://github.com/Obscyra-app/EmbeddingGemma2Swift)
- [Ready-to-use weights](https://huggingface.co/Nurymanau/EmbeddingGemma-2-MLX-Swift)
- [Measured validation and limitations](VALIDATION.md)

This is a developer SDK and CLI with a small iOS validation demo. It is not a
chat model or a standalone photo-library application. Inference runs locally.
Download weights explicitly; the Swift runtime never silently downloads them.

```sh
git clone https://github.com/Obscyra-app/EmbeddingGemma2Swift.git
cd EmbeddingGemma2Swift
python3 scripts/download_release.py --output models/release
xcodebuild -scheme eg2-embed -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .build-xcode build
```

Create `request.json`:

```json
{"query":"How can I check a downloaded file?","documents":["Compare its SHA-256 checksum with the source.","Tomorrow will be sunny."]}
```

```sh
.build-xcode/Build/Products/Release/eg2-embed models/release/text-q8 request.json
```

The downloader verifies every file against pinned SHA256 values. Text+vision
is the default download (~514 MB including tokenizer and metadata); add
`--audio` for the optional BF16 audio tower (~611 MB). No account is needed.
Python is only needed for this setup helper and reference tooling, not for
Swift inference; weights can also be downloaded manually from Hugging Face.

Add to a Swift package:

```swift
.package(url: "https://github.com/Obscyra-app/EmbeddingGemma2Swift.git", from: "0.1.0")
// Target dependency:
.product(name: "EmbeddingGemma2", package: "EmbeddingGemma2Swift")
```

Build applications with Xcode so that MLX Metal resources are included.

## Dependencies and model

- MLX Swift **0.32.3**, Swift Transformers **1.3.0**; transitive revisions are in `Package.resolved`.
- Swift 6.3 / Xcode; Apple Silicon, macOS 14+ / iOS 17+ declared package minimums. Minimum-OS execution requires device validation.
- Google checkpoint `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- BF16/FP32, or this package’s versioned affine 8/4-bit text/vision checkpoints. FP16 is rejected. Audio weights remain BF16/FP32.

The reference/source preparation scripts use Python, PyTorch, Transformers and safetensors. They are development tools, not application dependencies. `download_source.py` downloads pinned public original weights and verifies known LFS SHA256. `prepare_reference.py` extracts the text tensors **unchanged**, preserving the original checkpoint, and produces reference vectors using Google's upstream Transformers implementation on CPU FP32. It also produces a per-file SHA256 manifest. Use `--output` for the download destination and `--source`, `--output`, `--results` for extraction/reference destinations. `--extract-only` skips reference execution. Defaults use the workspace `models/` and `runs/` layout.

## Native API

```swift
import EmbeddingGemma2
let encoder = try await TextEncoder(directory: localModelDirectory)
let query = try await encoder.encode(["Как проверить контрольную сумму?"], task: .search)
let docs = try await encoder.encode(["Сравните SHA-256 файла с эталоном."], task: .document)
let score = zip(query[0], docs[0]).reduce(Float(0)) { $0 + $1.0 * $1.1 }
```

`TextEncoder` serializes execution in its own actor. Pass a local model directory containing `config.json`, `model.safetensors`, `tokenizer.json` and `tokenizer_config.json`. Runtime does not download weights. Use `.raw` only for already formatted inputs; prefixes must be applied once. To include a title, format with `EmbeddingTask.document.format(text, title: title)` and encode with `.raw`.

Supported output dimensions: 128, 256, 512, 768; vectors are re-normalized after truncation. The dimension must match for queries/documents. Inputs over 8192 tokens are rejected, not silently truncated. Chunk long documents explicitly. The runtime checks token range, masks, tensor names/shapes and weight precision. The text API rejects media placeholder IDs; use the image/audio APIs for media.

## Images and cross-modal search

```swift
// Keep the existing text checkpoint, and load the separately extracted vision weights.
try await encoder.enableVision(directory: localVisionDirectory)
let picture = try await encoder.encodeImage(url: imageURL, maxSoftTokens: 280)
let descriptions = try await encoder.encode(["красный круг", "синий квадрат"], task: .search)
let scores = descriptions.map { zip(picture, $0).reduce(Float(0)) { $0 + $1.0 * $1.1 } }
```

Image vectors share the text embedding space: rank images by text queries, or
rank text descriptions by an image. Image-only inputs receive no text prefix.
Each image is encoded separately; budgets 70, 140 and 280 are supported. A
larger budget changes the resize and representation, not just output length.
MRL dimensions are a separate option (`dimensions: 128`, etc.).

`ImageProcessor` implements the pinned aspect-ratio sizing and Pillow 12.0
bicubic algorithm. RGB8 resized bytes, patches and positions exactly matched
the reference for the tested PNG fixtures, all eight EXIF orientations and
grayscale. `RGBImage.load` uses ImageIO, converts to sRGB and applies EXIF.
JPEG/HEIC and embedded color profiles can decode differently across libraries;
exact decoder parity for these formats has not been established. Use the
`RGBImage(width:height:bytes:)` API for standardized pixels. Transparent images
are rejected; flatten them explicitly to an opaque background. Decode is
limited to 40 megapixels and 20000 pixels per side.

Extract vision weights and generate independent reference fixtures:

```sh
python scripts/prepare_vision_reference.py --source /path/to/original \
  --output /path/to/vision-bf16 --results /path/to/vision-reference
```

Vision extraction preserves all 167,757,824 parameters in 211 tensors,
including the bridge; it does not alter the original or text checkpoint.
The vision directory contains `config.json` and `vision.safetensors`.

Image CLI request:

```json
{"images":["/path/to/a.png","/path/to/b.png"],"vision_directory":"/path/to/vision-bf16","query":"красный круг","max_soft_tokens":70,"dimensions":768}
```

`query` is optional. Output contains image vectors, image-token IDs,
preprocessing metadata, timings and optionally ranked image paths.
`--preprocess-only` inspects the first image without loading the model;
an optional `pixel_reference` safetensors file checks pixels/positions.
`scripts/audit_vision.py REFERENCE ACTUAL --output AUDIT` verifies vectors,
token IDs, exact resized RGB hash and normalization. The image fixtures use
the same fixed FP32/BF16 tolerances as text.

## Audio and cross-modal search

```swift
try await encoder.enableAudio(directory: localAudioDirectory)
let sound = try await encoder.encodeAudio(url: wavURL, dimensions: 768)
let queries = try await encoder.encode(["звук двигателя", "пение птиц"], task: .search)
let scores = queries.map { zip(sound, $0).reduce(Float(0)) { $0 + $1.0 * $1.1 } }
await encoder.disableAudio() // Release the audio tower when no longer needed.
```

Use mono 16 kHz PCM16 RIFF WAV, or `AudioProcessor.prepare(samples:sampleRate:)`
with finite Float samples in [-1,1]. Supported length is 161–480000 samples
(up to 30 seconds); resampling/downmixing is the caller's responsibility.
Stereo, other sample rates, compressed audio and out-of-range inputs are
rejected explicitly. The model produces an embedding for the whole clip,
not a transcription or synthesized audio. Output dimensions are 128/256/512/768.
Text queries use `.search`; audio embeddings receive audio special tokens
without a text prefix. Audio→text and text→audio comparisons use cosine similarity.

The frontend implements the pinned periodic Hann window, 512-point FFT,
128 HTK mel bins, log floor, frame masking and padding. It is checked
separately from the Conformer and the final text-space projection.
Four synthetic silence/tone/noise/chirp fixtures passed Mac FP32/BF16 vector
thresholds; frontend max error was below 5e-7 with exact frame masks.
The same four audio vectors passed on two physical iPhone launches and
exactly matched Mac BF16. Repeat iPhone encode times were 0.050–0.143 s for
0.10–5.12-second signals; the first short clip took 2.367 s on initial launch.
These are narrow observations including frontend work, not a hardware benchmark.
A Russian description smoke test retrieved the intended signal in only 1/4
cases in each direction. Numerical agreement does not establish useful
real-world sound recognition quality. Frontend-only length edges were checked
through 30 seconds; full 30-second inference was not validated. Use `disableVision()` before loading audio to avoid
keeping both media towers resident when they are not needed.

```sh
python scripts/prepare_audio_reference.py --source /path/to/original \
  --output /path/to/audio-bf16 --results /path/to/audio-reference
```

This extracts 752 unchanged tensors (305,611,520 elements, including clip
bounds) into `audio.safetensors`, with `config.json` and a SHA256 manifest.
CLI request:

```json
{"audio":["/path/to/clip.wav"],"audio_directory":"/path/to/audio-bf16","query":"пение птиц","dimensions":768}
```

`query` is optional. Output includes vectors, token IDs, sample/frame/token
counts, timings and ranked paths. `--audio-preprocess-only` inspects the first
clip without loading model weights; `feature_reference` can point to a saved
frontend safetensors reference. `scripts/audit_audio.py REFERENCE ACTUAL
--output AUDIT` independently checks frontend error (<=2e-5), metadata,
tokenization and final vectors against the official CPU FP32 implementation.

## Quantized text and vision

The selected profile is **8-bit affine, group64** with BF16 activations.
All compatible 2D weight matrices are packed, including token embeddings;
normalization/scalars and the 3D vision position table remain unquantized.
Linear layers use native MLX packed matrix multiplication. Token lookup
unpacks only selected rows, never the full vocabulary table.

This is standard MLX post-training quantization, not a new compression method
or a retrained Google model. Use new destination folders:

```sh
eg2-embed --quantize /path/to/text-bf16 /path/to/text-q8 8
eg2-embed --quantize /path/to/vision-bf16 /path/to/vision-q8 8
eg2-embed /path/to/text-q8 request-with-vision-q8.json
```

The converter refuses existing destinations and quantized sources. It writes
`quantization.json`, source/output SHA256 and packed weights atomically.
This versioned layout is specific to this package; arbitrary MLX-VLM
checkpoints are not automatically compatible. Keep each file with its
metadata. The API is unchanged; point `TextEncoder` and `enableVision` at
the converted directories. CLI `precision` describes working activations;
`text_weight_format` separately identifies packed weights.

Measured weight-file sizes (tokenizer/config files excluded):

| Profile | Text | Vision | Combined |
|---|---:|---:|---:|
| Original BF16 | 542,056,416 B | 335,543,232 B | 877,599,648 B |
| Selected 8-bit | 288,118,885 B | 193,095,171 B | 481,214,056 B |
| Experimental 4-bit | 152,656,118 B | 117,105,857 B | 269,761,975 B |

8-bit passed the fixed dev gate (minimum cosine .995, at most one lost
correct text top1, synthetic shape retrieval preserved). Both candidates
scored 24/24 on the simple bilingual text dev; 4-bit nevertheless failed the
vector-drift gate (minimum cosine .97396). A separate held-out synthetic
set retained 16/16 text results and 6/6 text→image results with q8. Two
physical iPhone launches passed the original vector tolerances; text/image
vectors matched Mac q8, while one audio fixture differed within tolerance.
4-bit remains available for experiments but is not the selected profile. These synthetic checks do not establish
broad retrieval quality. The separate FP32 kernel audit uses independent
NumPy unpacking and the official PyTorch model, with the original strict
cosine/error thresholds, for both 4-bit and 8-bit.

## CLI

```sh
xcodebuild -scheme eg2-embed -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build-xcode build
# Locate the resulting executable under .build-xcode/Build/Products/Release.
eg2-embed /path/to/text-bf16 request.json
```

`swift build` alone does not compile MLX's Metal shaders; use Xcode for runnable binaries.

```json
{"query":"Как проверить целостность файла?","documents":["Сравните SHA-256 с эталоном.","Прогноз погоды на завтра."],"dimensions":768}
```

Or `{"texts":["..."],"task":"document","dimensions":256,"repetitions":5}`. JSON output includes embeddings, tokens, load duration, encode durations and MLX memory counters; search requests include ranked documents. Encode duration includes tokenization, model execution and host output transfer; it must not be compared directly to forward-only benchmarks. `--float32` selects FP32 evaluation. For independent kernel checks, a request may contain rectangular `input_ids` and `attention_mask` arrays.

## Validation and scope

The CLI limits reusable MLX cache to 64 MiB; library callers choose their own policy.

`guard.py` wraps a local run with a timeout and +256 MiB system-swap watchdog, retaining stdout, stderr and a receipt. `audit.py` checks against the independently generated reference: FP32 cosine >=0.99999/max error <=0.0001; BF16 cosine >=0.999/max error <=0.01; unit norm, finite values and token identity are checked separately. The tests include local attention boundaries and bidirectionality.

These checks establish a bounded implementation comparison, not general retrieval quality or performance superiority. No IFStruct data is used. No cloud GPU is required for this stage.

To reproduce (from the package directory, with a Python 3.12 environment):

```sh
python -m pip install -r requirements-reference.lock
python scripts/download_source.py --output /path/to/original
python scripts/prepare_reference.py --source /path/to/original \
  --output /path/to/text-bf16 --results /path/to/reference
python scripts/validate_mac.py \
  --executable .build-xcode/Build/Products/Release/eg2-embed \
  --model /path/to/text-bf16 --reference /path/to/reference \
  --output /path/to/new-validation-directory
xcodebuild -scheme EmbeddingGemma2Swift-Package -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build-xcode test
```

Run numerical checks after builds finish. The validation harness includes
seven invalid-input checks and a padded-batch/single comparison. The
independent audit was also checked to reject a deliberately corrupted vector.
See [the iOS example](Examples/iOS/README.md) for physical-device validation.

Earlier unquantized text-only timings (2026-10-06; narrow fixtures, not a hardware comparison):

| Device / condition | Seven short inputs, padded length 52 | One 959-token input |
| --- | ---: | ---: |
| Mac M3, warm short batch (median of 5) / separate long run | 0.102 s | 0.290 s |
| iPhone 16 Pro Max, second app launch | 0.128 s | 0.380 s |

Load took about 1.28 s on Mac and 1.24 s on the second iPhone launch.
The first iPhone short batch took 3.776 s and is retained as a cold-start
observation. Mac successful BF16 runs sampled about 924–1299 MiB process
footprint; positive system swap growth was zero. iPhone allocator peak was
875 MiB on the repeat; total app footprint was not measured. A first Mac
attempt overlapping a build was stopped by the swap watchdog and is excluded
from the successful-run timings. Full context, sustained thermals, battery
use, older OS/device support and broad retrieval quality remain untested.

`scripts/package_source.py --output /path/to/source.zip` creates a local,
allowlisted review archive with SHA256 for every source file. It includes no
weights, fixtures, raw results, signing profiles, or build products and does
not publish anything.

## Attribution

Google DeepMind created the model and released the checkpoint under Apache-2.0. The native architecture implementation is informed by Google's Transformers implementation (Apache-2.0) and Prince Canuma / MLX-VLM's Python port (MIT), revision `3d87e88402f307efbf68e568971aa887ee7d9ed0`. Image resizing adapts Pillow 12.0's MIT-CMU implementation. Preserve `NOTICE`, `APACHE-2.0.txt`, `UPSTREAM_MLX_VLM_LICENSE` and `UPSTREAM_PILLOW_LICENSE` with distributions. MLX Swift and the tokenizer retain their own licenses. This is an independent Swift port, not a claim of first MLX or first iOS support.
