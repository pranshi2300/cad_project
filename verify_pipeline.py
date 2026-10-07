"""Canonical test entry point — redirects to empirical pipeline validation."""
import runpy

if __name__ == "__main__":
    runpy.run_module("verify_empirical_pipeline", run_name="__main__")
