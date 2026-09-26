"""Run the frozen finalizer for SD3 only, without changing the all-model job."""
from pathlib import Path
import hashlib,sys
release=Path(sys.argv[1]).resolve()
p=release/'finalize.py'
s=p.read_text()
# (the internal run also asserted the SHA-256 of finalize.py; removed because path anonymization changed its bytes)
assert s.count("for model in ['sd3','flux','qwen']:")==1
s=s.replace("for model in ['sd3','flux','qwen']:","for model in ['sd3']:")
s=s.replace("heldout_prediction/runtime/analysis.json","heldout_prediction/runtime/analysis_sd3_20260913.json")
sys.path.insert(0,str(release))
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
