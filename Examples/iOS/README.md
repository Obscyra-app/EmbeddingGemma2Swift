# iPhone validation example

This is a development app that runs the frozen text, image and audio numerical fixtures
and writes `Documents/validation.json`. It is not a general search UI.
Validated on a physical iPhone 16 Pro Max with iOS 27.0 on 2026-10-06:
text, image and audio fixtures passed on two launches. This does not establish compatibility
with every device supported by the package's declared iOS 17 minimum.

1. Prepare the local text checkpoint and reference fixtures using the package's
   `scripts/prepare_reference.py` (see the root README).
2. Copy the checkpoint directory into `Model/` here (or create a local symlink).
   Copy `multilingual-request.json`, `multilingual-reference.json`,
   `long-request.json`, `long-reference.json` into `Fixtures/`.
   Run `prepare_vision_reference.py`, place its vision checkpoint directory
   in `Vision/`, and copy its four PNGs and `*-reference.json` files into
   `Fixtures/`. Run `prepare_audio_reference.py`, put its checkpoint in `Audio/`, and copy
   its four WAVs and reference JSONs into `Fixtures/`. Checkpoint folders may
   be local symlinks.
   Optionally convert text and vision with `eg2-embed --quantize` and use
   the selected q8 folders for `Model/` and `Vision/`; keep audio BF16.
   The report records the text weight format separately from activation dtype.
3. Run `xcodegen generate` in this directory. The generated Xcode project,
   model, fixtures and build output are ignored by Git.
4. Open the generated project in Xcode, select your development team and an
   appropriate provisioning profile, then build/run on a physical Apple device.
   Enable automatic signing in Xcode if appropriate for your account. Signing
   identifiers and profiles must remain local.

The demo bundles local weights, so it does not download anything at runtime.
It automatically validates on launch; the button reruns it. Keep the app in
the foreground until the result appears. Loading/first inference may compile
GPU kernels. The report separates tower loading from each fixture’s encode time.
The current app also validates four image inputs after the text fixtures,
including default budget280, then unloads vision and validates four audio
clips. It sets the global MLX reusable cache limit to
64 MiB as an application policy; the library itself does not set this limit.

To retrieve the result from a paired device:

```sh
xcrun devicectl device copy from --device YOUR_DEVICE \
  --domain-type appDataContainer \
  --domain-identifier local.embeddinggemma2.validation \
  --source Documents/validation.json --destination validation.json
```

Numerical parity is checked against the official CPU FP32 reference, using
the same BF16 thresholds as on Mac. MLX memory counters describe allocator
memory, not total app footprint. Battery use, sustained thermals, the full
8192-token context and older iPhones/OS releases require separate testing.
