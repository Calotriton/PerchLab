"""Suppress a curated allowlist of known-benign third-party log noise.

TensorFlow, XLA and CUDA print a fixed set of harmless lines on every GPU run
(the oneDNN notice, the CPU-instruction notice, GPU-device creation, XLA/cuDNN
init, ptxas register-spill notes, and repeated ``Delay kernel timed out`` timing
warnings), and perch-hoplite / the model add two more (a numpy ``np.divide``
``UserWarning`` and a duplicate-eBird-class-list warning). None affect results.

These originate in native (C++) code and are written straight to the **stderr
file descriptor**, so Python's ``warnings``/``logging`` filters never see them.
We therefore filter at the fd level: fd 2 is replaced with a pipe, and a daemon
thread forwards every line *except* those matching the explicit benign
allowlist. Anything unrecognised — including a genuine future error — is passed
through untouched, so nothing real is ever hidden.

Set ``PERCHLAB_LOG_FILTER=0`` to disable all of this and see raw output.
"""

from __future__ import annotations

import os
import re
import threading

# Each pattern targets one specific benign message seen on a normal run. They are
# deliberately narrow so they cannot match an unexpected/real error line.
_BENIGN_PATTERNS: tuple[str, ...] = (
    r"All log messages before absl::InitializeLog\(\) is called are written to STDERR",
    r"oneDNN custom operations are on",
    r"This TensorFlow binary is optimized to use available CPU instructions",
    r"To enable the following instructions:",
    r"cpu_feature_guard",
    r"Created device /job:localhost.*device:GPU",
    r"XLA service 0x[0-9a-fA-F]+ initialized",
    r"StreamExecutor \[\d+\]:",
    r"disabling MLIR crash reproducer",
    r"Loaded cuDNN version",
    r"Delay kernel timed out",
    r"ptxas warning : Registers are spilled to local memory",
    r"spill stores, .* spill loads",
    r"Compiled cluster using XLA",
    r"Failed to load class list.*duplicate entries in class list",
    r"'where' used without 'out'",
    r"framed_audio = np\.divide",
)
_BENIGN_RE = re.compile("|".join(_BENIGN_PATTERNS))

_installed = False


def quiet_known_logs() -> None:
    """Install the log suppressions. Idempotent; safe to call once at startup."""
    global _installed
    if _installed or os.environ.get("PERCHLAB_LOG_FILTER") == "0":
        return
    _installed = True

    # Drop the bulk of C++ INFO chatter cheaply (0=all,1=hide INFO,2=+WARNING,
    # 3=+ERROR). Level 1 keeps WARNING and ERROR so real problems still print;
    # the fd filter below removes the few benign WARNING/ERROR lines by content.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    _install_stderr_filter()


def _install_stderr_filter() -> None:
    """Replace fd 2 with a filtered pipe; forward all non-benign lines."""
    try:
        real_fd = os.dup(2)
    except OSError:
        return  # No usable stderr (e.g. detached); nothing to filter.

    # Preserve interactive rendering: once fd 2 is a pipe, ``sys.stderr.isatty()``
    # is False and Rich would drop its live progress bar/colour. If the *real*
    # stderr is a terminal, tell Rich to keep terminal mode via FORCE_COLOR.
    if os.isatty(real_fd) and "FORCE_COLOR" not in os.environ and "NO_COLOR" not in os.environ:
        os.environ["FORCE_COLOR"] = "1"

    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def pump() -> None:
        buf = bytearray()
        while True:
            try:
                chunk = os.read(read_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            start = 0
            while True:
                nl = buf.find(b"\n", start)
                if nl == -1:
                    break
                line = bytes(buf[start : nl + 1])
                if _BENIGN_RE.search(line.decode("utf-8", "replace")) is None:
                    os.write(real_fd, line)
                start = nl + 1
            # Residual has no newline: native log lines are written whole (with
            # their newline), so leftover bytes are Rich's progress escapes —
            # forward them promptly so the live display keeps rendering.
            if start < len(buf):
                os.write(real_fd, bytes(buf[start:]))
            del buf[:]

    threading.Thread(target=pump, name="perchlab-stderr-filter", daemon=True).start()
