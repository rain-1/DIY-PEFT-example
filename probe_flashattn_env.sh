#!/usr/bin/env bash
set -euo pipefail

json_escape() {
  python3 - <<'PY'
import json,sys
print(json.dumps(sys.stdin.read()))
PY
}

cmd_out() {
  # prints stdout+stderr, never fails the whole script
  local cmd="$1"
  (bash -lc "$cmd" 2>&1 || true)
}

have() { command -v "$1" >/dev/null 2>&1; }

python_bin="${PYTHON_BIN:-python3}"
pip_bin="${PIP_BIN:-pip3}"

# Prefer venv/conda python if active
if have python; then python_bin="${python_bin:-python}"; fi
if have pip; then pip_bin="${pip_bin:-pip}"; fi

ts="$(date -Is 2>/dev/null || date)"

{
  echo "{"
  echo "  \"timestamp\": $(printf "%s" "$ts" | json_escape),"

  echo "  \"os_release\": $(cmd_out 'cat /etc/os-release' | json_escape),"
  echo "  \"kernel\": $(cmd_out 'uname -a' | json_escape),"
  echo "  \"glibc\": $(cmd_out 'ldd --version | head -n 1' | json_escape),"

  echo "  \"nvidia_smi\": $(cmd_out 'nvidia-smi' | json_escape),"
  echo "  \"nvidia_smi_q\": $(cmd_out 'nvidia-smi -q' | json_escape),"
  echo "  \"gpu_list\": $(cmd_out 'nvidia-smi --query-gpu=index,name,uuid,compute_cap,driver_version,memory.total --format=csv,noheader' | json_escape),"

  echo "  \"cuda\": {"
  echo "    \"nvcc\": $(cmd_out 'nvcc --version' | json_escape),"
  echo "    \"cuda_home\": $(cmd_out 'echo ${CUDA_HOME:-}' | json_escape),"
  echo "    \"ld_library_path\": $(cmd_out 'echo ${LD_LIBRARY_PATH:-}' | json_escape)"
  echo "  },"

  echo "  \"build_tools\": {"
  echo "    \"gcc\": $(cmd_out 'gcc --version | head -n 1' | json_escape),"
  echo "    \"gpp\": $(cmd_out 'g++ --version | head -n 1' | json_escape),"
  echo "    \"cmake\": $(cmd_out 'cmake --version | head -n 1' | json_escape),"
  echo "    \"ninja\": $(cmd_out 'ninja --version' | json_escape),"
  echo "    \"git\": $(cmd_out 'git --version' | json_escape),"
  echo "    \"make\": $(cmd_out 'make --version | head -n 1' | json_escape),"
  echo "    \"python_bin\": $(cmd_out "command -v $python_bin" | json_escape),"
  echo "    \"pip_bin\": $(cmd_out "command -v $pip_bin" | json_escape)"
  echo "  },"

  echo "  \"python\": {"
  echo "    \"python\": $(cmd_out "$python_bin -V" | json_escape),"
  echo "    \"which_python\": $(cmd_out "command -v $python_bin" | json_escape),"
  echo "    \"pip\": $(cmd_out "$pip_bin -V" | json_escape),"
  echo "    \"which_pip\": $(cmd_out "command -v $pip_bin" | json_escape),"
  echo "    \"venv\": $(cmd_out "$python_bin -c 'import sys; print(getattr(sys, \"prefix\", \"\")); print(getattr(sys, \"base_prefix\", \"\"));'" | json_escape),"
  echo "    \"site\": $(cmd_out "$python_bin -c 'import site,sys; print(\"\\n\".join(site.getsitepackages() if hasattr(site,\"getsitepackages\") else [])); print(site.getusersitepackages());'" | json_escape)"
  echo "  },"

  echo "  \"torch\": $(cmd_out "$python_bin - <<'PY'\nimport json\ntry:\n  import torch\n  d={\n    'torch': torch.__version__,\n    'cuda_available': torch.cuda.is_available(),\n    'torch_cuda_version': torch.version.cuda,\n    'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,\n    'cuda_device_name_0': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n    'capability_0': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,\n    'bf16_supported': (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),\n  }\n  print(json.dumps(d))\nexcept Exception as e:\n  print(json.dumps({'error': repr(e)}))\nPY" | json_escape),"

  echo "  \"installed_packages\": $(cmd_out "$pip_bin list --format=freeze | egrep -i '^(torch|triton|nvidia|flash-attn|xformers|transformers|accelerate|packaging|setuptools|wheel|pip)=' || true" | json_escape)"
  echo "}"
} | python3 -c "import sys,json; print(json.dumps(json.loads(sys.stdin.read()), indent=2))"

