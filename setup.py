"""Build script for stabkernel.

Project metadata lives in ``pyproject.toml`` (PEP 621). This file only handles
building the native shared library from the hand-written C + assembly sources
and bundling it inside the ``stabkernel`` package so the ctypes bindings can
load it.

Key design point: the shared library is compiled WITHOUT ``-march=native`` so a
wheel built on one machine runs everywhere. CPU features (AVX-512 / AVX2) are
selected at runtime via CPUID in ``src/dispatch.c`` — the hand-written kernels
in ``src/gf2_x86.S`` are only *called* after that check, so building on a
baseline CPU is safe.
"""
import os
import platform
import subprocess
import sys
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

ROOT = os.path.dirname(os.path.abspath(__file__))

C_SOURCES = [
    "src/gf2_scalar.c",
    "src/dispatch.c",
    "src/gf2_linalg.c",
    "src/wrappers.c",
    "src/gf2_parallel.c",
]
ASM_SOURCES_X86 = ["src/gf2_x86.S"]


def _is_x86():
    return platform.machine().lower() in (
        "x86_64", "amd64", "x64", "i386", "i486", "i586", "i686",
    )


class BinaryDistribution(Distribution):
    """Force a platform-specific (non-pure) wheel; we ship a compiled .so."""

    def has_ext_modules(self):  # noqa: D401
        return True


class BuildNativeLib(build_ext):
    """Compile the native shared library directly with the C compiler.

    We bypass the usual per-Extension compilation because the tree mixes C with
    GNU-assembler (.S) sources, which distutils does not handle uniformly.
    """

    def build_extensions(self):  # override: ignore declared placeholder ext
        self._build_native()

    # -- helpers ---------------------------------------------------------
    def _cc(self):
        return os.environ.get("CC", "clang" if sys.platform == "darwin" else "cc")

    def _lib_name(self):
        if sys.platform == "darwin":
            return "libstabkernel.dylib"
        if sys.platform.startswith("win"):
            return "stabkernel.dll"
        return "libstabkernel.so"

    def _detect_openmp(self, cc):
        """Return (cflags, ldflags) enabling OpenMP if the toolchain supports it."""
        src = "#include <omp.h>\nint main(){return omp_get_max_threads();}\n"
        candidates = [(["-fopenmp"], ["-fopenmp"])]
        if sys.platform == "darwin":
            # Homebrew libomp locations for Apple clang
            for prefix in ("/opt/homebrew", "/usr/local"):
                inc = os.path.join(prefix, "opt", "libomp", "include")
                lib = os.path.join(prefix, "opt", "libomp", "lib")
                if os.path.isdir(lib):
                    candidates.append((
                        ["-Xpreprocessor", "-fopenmp", "-I" + inc],
                        ["-L" + lib, "-lomp"],
                    ))
            candidates.append((["-Xpreprocessor", "-fopenmp"], ["-lomp"]))
        for cflags, ldflags in candidates:
            with tempfile.TemporaryDirectory() as tmp:
                cpath = os.path.join(tmp, "t.c")
                opath = os.path.join(tmp, "t.out")
                with open(cpath, "w") as fh:
                    fh.write(src)
                try:
                    subprocess.check_call(
                        [cc, *cflags, cpath, "-o", opath, *ldflags],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    print("stabkernel: OpenMP enabled via", cflags)
                    return cflags, ldflags
                except (subprocess.CalledProcessError, OSError):
                    continue
        print("stabkernel: WARNING — OpenMP not available; "
              "multi-core kernels will run single-threaded.")
        return [], []

    def _build_native(self):
        cc = self._cc()
        outdir = os.path.join(self.build_lib, "stabkernel")
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, self._lib_name())

        sources = [os.path.join(ROOT, s) for s in C_SOURCES]
        if _is_x86():
            sources += [os.path.join(ROOT, s) for s in ASM_SOURCES_X86]

        cflags = ["-O3", "-Wall", "-Iinclude", "-fPIC"]
        ldflags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
        omp_c, omp_l = self._detect_openmp(cc)
        cflags += omp_c
        ldflags += omp_l

        cmd = [cc, *cflags, *ldflags, *sources, "-o", out]
        print("stabkernel: building native library\n  " + " ".join(cmd))
        subprocess.check_call(cmd, cwd=ROOT)
        print("stabkernel: wrote", out)


setup(
    # A placeholder Extension makes setuptools treat this as a binary wheel and
    # triggers build_ext; BuildNativeLib.build_extensions ignores it and builds
    # the real shared library instead.
    ext_modules=[Extension("stabkernel._placeholder", sources=["src/wrappers.c"])],
    cmdclass={"build_ext": BuildNativeLib},
    distclass=BinaryDistribution,
)
