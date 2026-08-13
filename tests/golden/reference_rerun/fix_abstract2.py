"""Convert MATLAB `methods (Abstract)` blocks to concrete error stubs (Octave compat)."""
import re
import pathlib

for p in pathlib.Path(".").rglob("*.m"):
    text = p.read_text(errors="replace")
    lines = text.split("\n")
    out, i, changed = [], 0, False
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*methods\s*\(\s*Abstract\s*\)", line):
            changed = True
            indent = re.match(r"^(\s*)", line).group(1)
            out.append(indent + "methods  % Octave-compat: was methods(Abstract)")
            i += 1
            while i < len(lines):
                b = lines[i]
                if re.match(r"^\s*end\s*$", b):
                    out.append(b)
                    i += 1
                    break
                s = b.strip()
                if s and not s.startswith("%"):
                    sig = s.split("%")[0].strip().rstrip(";")
                    out.append(indent + "    function " + sig)
                    out.append(indent + "        error('abstract');")
                    out.append(indent + "    end")
                else:
                    out.append(b)
                i += 1
            continue
        out.append(line)
        i += 1
    if changed:
        p.write_text("\n".join(out))
        print("patched", p)
