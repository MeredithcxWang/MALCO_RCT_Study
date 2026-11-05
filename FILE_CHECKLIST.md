# Release packaging

Run `python scripts/package_release.py --output GitHub_Upload` using the project environment. It copies the approved code files and original synthetic sample, creates a SHA256 manifest, and does not upload anything.

Clinical data, manuscripts, credentials, generated outputs, model weights and installed dependencies are excluded.
