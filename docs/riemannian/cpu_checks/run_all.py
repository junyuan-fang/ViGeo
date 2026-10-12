"""
Run every CPU check (pure Python standard library; no torch / numpy / GPU / checkpoints).

    python docs/riemannian/cpu_checks/run_all.py          # ~4 min, R2 dominates
    python docs/riemannian/cpu_checks/run_all.py --quick  # skip R2 (~20 s)

Results and interpretation: docs/riemannian/cpu_checks/RESULTS.md
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = [
    ('R1/R0 loss math', 'check_r1_losses.py'),
    ('R3 normal x pose', 'check_r3_normal_pose.py'),
    ('R4 redundancy', 'check_r4_redundancy.py'),
    ('R2 smoothing upper bound', 'check_r2_smoothing.py'),
]

if __name__ == '__main__':
    quick = '--quick' in sys.argv
    status = 0
    for name, script in CHECKS:
        if quick and script.startswith('check_r2'):
            print(f'\n##### {name}: skipped (--quick)')
            continue
        print(f'\n##### {name} ({script})')
        t0 = time.time()
        rc = subprocess.call([sys.executable, os.path.join(HERE, script)], cwd=HERE)
        print(f'##### {name}: exit {rc}, {time.time() - t0:.0f}s')
        status |= rc
    sys.exit(status)
