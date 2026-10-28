"""test_part9.py : checks the logic behind the dashboard and demo. Run: python test_part9.py"""
import py_compile
import subprocess
import sys
import numpy as np
from runner import SCHEMES, run_many, run_scheme, summarize_choices

print("1. Every scheme runs on every scenario")
for scenario in ("random", "bursty", "changing"):
    frames = 900 if scenario == "changing" else 60
    df, choices = run_many(SCHEMES, scenario, 1e-3, 20, 1500, frames, 1)
    assert list(df.index) == SCHEMES and (df["Throughput"] >= 0).all()
    assert choices is not None and len(choices) > 0
    print(f"   {scenario:<9} ok   proposed throughput {df.loc['PROPOSED ADAPTIVE', 'Throughput']:.3f}")

print("2. Same inputs give the same numbers (fixed seeds)")
a, _ = run_scheme("PROPOSED ADAPTIVE", "random", 1e-3, 20, 1500, 80, 3)
b, _ = run_scheme("PROPOSED ADAPTIVE", "random", 1e-3, 20, 1500, 80, 3)
assert a == b
print("   identical")

print("3. Sanity: on a clean channel everything works, on a very noisy one plain CRC gives up")
clean, _ = run_many(SCHEMES, "random", 1e-5, 20, 1500, 80, 1)
noisy, _ = run_many(["SR + CRC", "PROPOSED ADAPTIVE"], "random", 1e-2, 20, 1500, 80, 1)
assert (clean["Finished"] == 1).all() and clean.loc["SR + CRC", "Throughput"] > 0.8
assert noisy.loc["PROPOSED ADAPTIVE", "Throughput"] > 10 * max(noisy.loc["SR + CRC", "Throughput"], 1e-3)
print(f"   clean SR+CRC {clean.loc['SR + CRC', 'Throughput']:.3f} | noisy SR+CRC {noisy.loc['SR + CRC', 'Throughput']:.3f} "
      f"vs proposed {noisy.loc['PROPOSED ADAPTIVE', 'Throughput']:.3f}")

print("4. Command line demo runs")
out = subprocess.run([sys.executable, "demo.py", "--channel", "bursty", "--ber", "1e-3", "--frames", "60"],
                     capture_output=True, text=True)
assert out.returncode == 0 and "PROPOSED ADAPTIVE" in out.stdout
print("   ok")

print("5. Dashboard file has valid syntax (it needs streamlit installed to actually run)")
py_compile.compile("app.py", doraise=True)
print("   ok")
print("\nAll checks passed.")
