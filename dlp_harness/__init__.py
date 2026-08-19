############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_harness: measurement + validation suite for the DLP
# subsystem (backend/app/services/dlp_scanner.py + worker).
#
# Treats DLP as a data-science problem: synthetic labeled
# corpora, offline scanner evaluation (precision/recall/F1/
# specificity, threshold sweeps), end-to-end detection
# through the gateway, load/overhead matrices, and a report
# generator. See dlp_harness/README.md and TESTING.md.
#
############################################################

"""MindRouter DLP evaluation harness."""

__version__ = "0.1.0"
