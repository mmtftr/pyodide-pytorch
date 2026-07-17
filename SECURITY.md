# Security policy

This project publishes unofficial, CPU-only PyTorch wheels for Pyodide. It
does not modify PyTorch's security support policy and does not make an old
PyTorch or Pyodide release safe for untrusted workloads.

Please report vulnerabilities in these build scripts privately through
GitHub's security-advisory interface. Report PyTorch vulnerabilities to the
PyTorch project and Pyodide vulnerabilities to the Pyodide project.

Release artifacts are accompanied by SHA-256 checksums and GitHub artifact
attestations. Consumers should pin a release asset by digest.
