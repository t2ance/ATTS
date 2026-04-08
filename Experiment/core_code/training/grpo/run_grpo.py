"""Entrypoint for ATTS GRPO training.

Registers the ATTSAgentLoop before verl starts, then delegates to verl.
"""

import training.grpo.atts_agent_loop  # noqa: F401 -- registers atts_agent
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
