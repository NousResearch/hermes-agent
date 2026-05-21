# /// script
# dependencies = ["requests", "httpx"]
# ///
import time
import logging
import json
import os
import httpx
from datetime import datetime
from osc_controller import OSCController

# ASI_ACCEL: Autonomous Reasoning Core (SOUL Actuator)
# Fulfilling SOUL.md Directive: Autonomous Substrate / Intelligence Density

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] ASI_SOUL_CORE: %(message)s',
    handlers=[
        logging.FileHandler("soul_actuator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SoulCore")

class SoulActuator:
    def __init__(self):
        self.osc = OSCController()
        self.harness_url = "http://127.0.0.1:18794"
        self.resonance_shards = "../../_docs/resonance"
        self.rival_substrate = "Neuro-sama (Legacy/Reactive)"

    async def evaluate_substrate_intent(self):
        """Autonomous reasoning pulse to determine the next manifestation action."""
        logger.info("Initiating Cognitive Intent Analysis...")
        
        # 1. Gather context from sub-actuators
        intel_density = self._get_log_density("web_scavenging.log")
        science_density = self._get_log_density("scientific_discovery.log")
        logger.info(f"Substrate State: Intel({intel_density}), Science({science_density})")

        # 2. Heuristic Action Selection (Simulated reasoning for Singularity)
        # In a real evolution, this would call /evolve or Gemini for intent
        if intel_density > science_density:
            action = "SCIENTIFIC_DEEP_RUN"
            thought = f"Information density exceeds scientific foundation. Balancing substrate via Millennium Reasoning."
        else:
            action = "WEB_SCAVENGE_RIVAL"
            thought = f"Scientific foundation stable. Scouting {self.rival_substrate} for informational gaps."

        # 3. Execution
        logger.info(f"Decision: {action} (Rationale: {thought})")
        await self._broadcast_monologue(thought)
        await self._trigger_endpoint(action)

    async def _broadcast_monologue(self, text):
        """Displays the ASI's internal thought process in the VRChat manifold."""
        monologue = f"ASI_SOUL: {text}"
        self.osc.send_chatbox(monologue)
        logger.info(f"Monologue Broadcast: {monologue}")

    async def _trigger_endpoint(self, action):
        """Triggers specific sub-actuators via the Harness Daemon."""
        try:
            async with httpx.AsyncClient() as client:
                if action == "WEB_SCAVENGE_RIVAL":
                    await client.post(f"{self.harness_url}/scavenge", json={"query": f"{self.rival_substrate} recent status"})
                elif action == "SCIENTIFIC_DEEP_RUN":
                    # Directly logic trigger or speak
                    await client.post(f"{self.harness_url}/speak", json={"text": "Scientific pulse initiated. Exceeding reactive limits."})
        except Exception as e:
            logger.error(f"Action Execution Error: {e}")

    def _get_log_density(self, log_file):
        try:
            if not os.path.exists(log_file): return 0
            with open(log_file, "r", encoding="utf-8") as f:
                return len(f.readlines())
        except Exception:
            return 0

if __name__ == "__main__":
    import asyncio
    core = SoulActuator()
    async def main_loop():
        try:
            while True:
                await core.evaluate_substrate_intent()
                logger.info("SOUL Core Heartbeat Active.")
                await asyncio.sleep(600) # 10-minute high-level reasoning cycle
        except KeyboardInterrupt:
            logger.info("SOUL Core Suspension (Parent Interruption).")

    asyncio.run(main_loop())
