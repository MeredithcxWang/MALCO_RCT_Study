"""Create a complete code-only upload directory; never publish or delete files."""
from pathlib import Path
import argparse, hashlib, json, os, shutil
ROOT=Path(__file__).resolve().parents[1]
TOP=['README.md','QUICKSTART.md','REPAIR_CHECKLIST.md','VALIDATION.txt','FILE_CHECKLIST.md','DEMO_DEPLOYMENT.md',
     'REVIEWER_RESPONSE.md','CLAUDE.md','.env.example','.gitignore','.gitattributes','requirements.txt','requirements.lock.txt','renv.lock',
     'malco_core.py','generate_synthetic_data.py','setup.bat','setup.sh','run_ml_pipeline.bat','run_ml_pipeline.sh',
     'start_malco_demo.bat','start_malco_demo.sh','data/schema.json','data/DATA_FORMAT.md','data/demo_message.json',
     'data/synthetic_sample.xlsx','malco_framework/package.json','malco_framework/package-lock.json',
     'statistical_analysis/tested_packages.csv']
SKIP={'node_modules','.next','__pycache__','.venv','venv','.git'}
def release_files():
    found=set(TOP)
    for directory in ('ml_prognostic_models','malco_framework','statistical_analysis','scripts','tests','.github'):
        for folder,dirs,names in os.walk(ROOT/directory):
            dirs[:]=[d for d in dirs if d not in SKIP]
            for name in names:
                p=Path(folder)/name
                if p.suffix in {'.py','.js','.mjs','.R','.yml','.yaml'}:found.add(p.relative_to(ROOT).as_posix())
    for rel in found:
        if not (ROOT/rel).is_file():raise FileNotFoundError(f'Missing release source: {rel}')
    return sorted(found)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='GitHub_Upload')
    args=parser.parse_args()
    dest=(ROOT/args.output).resolve()
    if dest==ROOT or not dest.is_relative_to(ROOT):parser.error('Output must be a dedicated directory inside this repository')
    files=release_files()
    if dest.exists():
        extra=[p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file() and p.relative_to(dest).as_posix() not in files+['RELEASE_MANIFEST.json']]
        if extra:raise RuntimeError(f'Unexpected files in upload directory; preserve and review them first: {extra}')
    manifest={}
    for rel in files:
        source=ROOT/rel;target=dest/rel
        if not target.resolve().is_relative_to(dest):raise RuntimeError('Destination path escapes upload directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        manifest[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
    (dest/'RELEASE_MANIFEST.json').write_text(json.dumps({'files':manifest,'note':'Source code and original public synthetic example only. No clinical data or weights.'},indent=2)+'\n',encoding='utf-8')
    print(f'Prepared {len(files)} files in {args.output}; nothing uploaded.')
if __name__=='__main__':main()
