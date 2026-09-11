#!/usr/bin/env bash
# Apply the NPU compatibility patches to the installed sglang package.
# Run once per environment (idempotent guards inside each patch's targets
# make accidental re-application harmless: the diffs would simply fail).
#
# Usage: bash patches/npu/apply_npu_patches.sh [SITE_PACKAGES_DIR]
set -euo pipefail

SITE="${1:-$(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')}/.."
SITE="$(cd "$SITE" && pwd)"
echo "Applying NPU patches under $SITE"

patch -N --dry-run -s -p1 \
  "$SITE/sglang/srt/models/moss_vl.py" < \
  "$(dirname "$0")/0001-fix-vision-rope-for-transformers-5-and-npu-inv-freq.patch" \
  && patch -N -p1 "$SITE/sglang/srt/models/moss_vl.py" < \
     "$(dirname "$0")/0001-fix-vision-rope-for-transformers-5-and-npu-inv-freq.patch" \
  || echo "0001 already applied or incompatible"

patch -N --dry-run -s -p1 \
  "$SITE/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py" < \
  "$(dirname "$0")/0002-fix-cross-attention-extend-sdpa-alignment.patch" \
  && patch -N -p1 "$SITE/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py" < \
     "$(dirname "$0")/0002-fix-cross-attention-extend-sdpa-alignment.patch" \
  || echo "0002 already applied or incompatible"

python - <<'EOF'
import ast
import sglang, os
root = os.path.dirname(sglang.__file__)
for rel in ("srt/models/moss_vl.py",
            "srt/hardware_backend/npu/attention/ascend_torch_native_backend.py"):
    ast.parse(open(os.path.join(root, rel)).read())
    print("verified:", rel)
EOF
echo "NPU patches applied."
