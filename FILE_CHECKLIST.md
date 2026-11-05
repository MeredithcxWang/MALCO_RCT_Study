# Release packaging

Run `python scripts/package_release.py --output GitHub_Upload` to assemble the complete reviewed code package. The command uses an explicit allowlist and writes RELEASE_MANIFEST.json with SHA256 values. It does not upload or push anything.

Real validation data, manuscript DOCX files, .env credentials, node_modules, .venv, .next, generated model weights and output data are excluded. The original public synthetic example is the only XLSX included.
