from __future__ import annotations

import os
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalNativeBuildExt(build_ext):
    """Build C++20 OpenMP extension; retry without OpenMP if needed."""

    def build_extensions(self):
        is_msvc = self.compiler.compiler_type == "msvc"
        for ext in self.extensions:
            if is_msvc:
                ext.extra_compile_args = ["/O2", "/std:c++20", "/openmp"]
                ext.extra_link_args = []
            else:
                ext.extra_compile_args = ["-O3", "-std=c++20", "-fopenmp", "-fvisibility=hidden"]
                ext.extra_link_args = ["-fopenmp"]
        try:
            super().build_extensions()
        except Exception:
            # Retry without OpenMP on toolchains lacking libgomp/libomp headers.
            for ext in self.extensions:
                if not is_msvc:
                    ext.extra_compile_args = ["-O3", "-std=c++20", "-fvisibility=hidden"]
                    ext.extra_link_args = []
            super().build_extensions()


ext_modules = []
if os.environ.get("STRATA_DISABLE_NATIVE") != "1":
    ext_modules.append(
        Extension(
            "strata._native",
            sources=["src/strata/_native.cpp"],
            language="c++",
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": OptionalNativeBuildExt},
)
