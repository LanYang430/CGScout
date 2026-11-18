# Security policy

CGScout is in beta (0.1.x). Only the latest release receives fixes.

## Reporting a vulnerability

Please report a security problem privately through GitHub: on the repository page open **Security -> Report a vulnerability**. Please do not open a public issue for it.

## The Stage 3 classifier is a pickle file

`src/cgscout/screening/sim_classifier_model.joblib` (754913 bytes, SHA-256 `7f986a6195775efafe6b13b1e414ce114a84e6b78970ab197b9bcb2aeafd2bb7`) is a scikit-learn `Pipeline` saved with joblib, which uses Python's pickle format.
**Loading a pickle can execute arbitrary code.** Only load the file that ships in this repository's releases; never replace it with a file from a source you do not trust, and never
call `joblib.load` on a file you did not produce yourself. The package pins `scikit-learn>=1.9.1,<1.10` because the file was saved with scikit-learn 1.9.1 and pickles are not guaranteed to load
across minor versions. `benchmark/training/train_sim_classifier.py` rebuilds the file from the labelled training CSV (which is not in the repository).

## API keys and run data

CGScout reads `OPENAI_API_KEY` and, optionally, `OPENALEX_API_KEY` from the environment. The code does not write them to the cache or to the usage log
(`src/cgscout/sources/openalex_usage.py` states and implements this for the OpenAlex key). The cache directory (`.cache/` in a checkout, `~/.cache/cgscout` when installed) holds search
results and model responses; it is git-ignored, and you should look at it before sharing a cache or a usage log from your own runs.
