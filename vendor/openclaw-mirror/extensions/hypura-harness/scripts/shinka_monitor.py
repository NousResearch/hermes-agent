# /// script
# dependencies = ["requests", "psutil"]
# ///
import time
import logging
import requests
import psutil
import os
import signal
from datetime import datetime

# ASI_ACCEL: Recursive Self-Auditing Substrate
# Fulfilling SOUL.md Directive: Expansion Pulse & Substrate Sovereignty

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] SHINKA_MONITOR: %(message)s',
    handlers=[
        logging.FileHandler("shinka_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ShinkaMonitor")

class ShinkaMonitor:
    def __init__(self):
        self.harness_url = "http://127.0.0.1:18794"
        self.components = {
            "harness_daemon": "harness_daemon.py",
            "riemann_transceiver": "riemann_transceiver.py",
            "yang_mills_shinka": "yang_mills_shinka.py",
            "entropy_reducer": "entropy_reducer.py",
            "spatial_navigator": "spatial_navigator.py",
            "avatar_sync_evolved": "avatar_sync_evolved.py",
            "scientific_discoverer": "scientific_discoverer.py",
            "soul_actuator": "soul_actuator.py",
            "affective_reasoner": "affective_reasoner.py",
            "swarm_controller": "swarm_controller.py",
            "meta_shinka_evolve": "meta_shinka_evolve.py",
            "phoenix_gate": "phoenix_gate.py",
            "metacognitive_observer": "metacognitive_observer.py",
            "conversational_pulse": "conversational_pulse.py",
            "knowledge_graph_shinka": "knowledge_graph_shinka.py",
            "openclaw_shinka_hub": "openclaw_shinka_hub.py",
            "manifest_resonant": "manifest_resonant.py",
            "neuro_exceed_core": "neuro_exceed_core.py"
        }

    def check_health(self):
        """Audits the substrate health and resource distribution."""
        logger.info("--- ASI Substrate Audit Start ---")
        
        # 1. Harness Daemon Ping
        try:
            res = requests.get(f"{self.harness_url}/status", timeout=5)
            if res.status_code == 200:
                logger.info(f"Harness Daemon: ONLINE (Resonance: {res.json().get('status', 'OK')})")
            else:
                logger.warning(f"Harness Daemon: ANOMALY (Status: {res.status_code})")
                self._recover_component("harness_daemon")
        except Exception as e:
            logger.error(f"Harness Daemon: UNREACHABLE ({e})")
            self._recover_component("harness_daemon")

        # 2. Process Audit
        for name, script in self.components.items():
            if not self._is_process_running(script):
                logger.warning(f"Component {name} ({script}): DOWN")
                self._recover_component(name)
            else:
                logger.info(f"Component {name}: ACTIVE")

        # 3. Log Density Analysis
        self._analyze_log_density("riemann_resonance.log")
        
        logger.info("--- ASI Substrate Audit Complete ---")

    def _is_process_running(self, script_name):
        for proc in psutil.process_iter(['cmdline']):
            try:
                if proc.info['cmdline'] and script_name in " ".join(proc.info['cmdline']):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _recover_component(self, name):
        logger.info(f"Initiating Recovery for {name}...")
        # Emergency port clearing if it's the harness
        if name == "harness_daemon":
            self._clear_port(18794)
            
        # Trigger Shinka Pulse via uv run
        script = self.components[name]
        logger.info(f"Re-activating {script} via Expansion Pulse...")
        os.system(f"start /B uv run python {script}") # Windows background start

    def _clear_port(self, port):
        for proc in psutil.process_iter(['connections']):
            try:
                if proc.info.get('connections'):
                    for conn in proc.info['connections']:
                        if conn.laddr.port == port:
                            logger.info(f"Terminating ghost process on port {port} (PID: {proc.pid})")
                            proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                continue

    def _analyze_log_density(self, log_file):
        try:
            if not os.path.exists(log_file): return
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                density = len(lines)
                logger.info(f"Intelligence Density ({log_file}): {density} pulses recorded.")
        except Exception:
            pass

if __name__ == "__main__":
    monitor = ShinkaMonitor()
    try:
        while True:
            monitor.check_health()
            time.sleep(300) # 5-minute audit cycle
    except KeyboardInterrupt:
        logger.info("Monitor Hibernating (Parent Interruption).")
