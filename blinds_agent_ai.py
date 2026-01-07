#!/usr/bin/env python3
"""
WindowBlindsAgentAI - blinds agent with AI.

Funkcje:
- Pobiera stan środowiska z symulatora
- Używa AI do podejmowania decyzji o stanie rolet
- Steruje roletami w każdym pokoju
- Obsługuje awarie zasilania, światło dzienne, ochronę przed upałem
"""

import asyncio
import aiohttp
import logging
import sys
import os
import json
from typing import Optional

# Import Ollama client
try:
    import httpx
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("httpx library not available. Install with: pip install httpx")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WindowBlindsAgentAI")

logging.getLogger("httpx").setLevel(logging.WARNING)


class BlindsAgentAI:
    """Blinds agent with AI (Ollama)."""
    
    def __init__(self, simulator_url: str = "http://localhost:8080", 
                 ollama_base_url: str = "http://localhost:11434",
                 use_ai: bool = True,
                 model: str = "llama2"):
        if not use_ai:
            raise ValueError("AI is required. WindowBlindsAgentAI cannot work without AI.")
        self.simulator_url = simulator_url
        self.session = None
        self.running = False
        self.model = model
        self.ollama_base_url = ollama_base_url
        
        # Configuration
        self.poll_interval = 2.0  # sekundy
        
        # AI Configuration
        if use_ai and OLLAMA_AVAILABLE:
            self.use_ai = True
            logger.info(f"AI enabled: Ollama, model: {model}, base_url: {ollama_base_url}")
        else:
            self.use_ai = False
            if not OLLAMA_AVAILABLE:
                logger.error("httpx library not available. Install with: pip install httpx")
                raise RuntimeError("AI is required but httpx is not available. Install with: pip install httpx")
            if not use_ai:
                logger.error("AI is disabled. Agent cannot function without AI.")
                raise RuntimeError("AI is required. Do not use --no-ai flag.")
    
    async def start(self):
        """Starts the agent."""
        self.session = aiohttp.ClientSession()
        self.running = True
        logger.info(f"WindowBlindsAgentAI started - {self.simulator_url}")
        
        try:
            while self.running:
                await self.run_cycle()
                await asyncio.sleep(self.poll_interval)
        finally:
            await self.session.close()
    
    def stop(self):
        """Stops the agent."""
        self.running = False
        logger.info("WindowBlindsAgentAI stopped")
    
    async def get_state(self) -> dict | None:
        """Gets environment state from simulator."""
        try:
            async with self.session.get(f"{self.simulator_url}/api/environment/state") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Error getting state: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Error connecting: {e}")
            return None
    
    async def set_blinds(self, blinds_id: str, state: str) -> bool:
        """Sets blinds state."""
        try:
            payload = {"state": state}
            url = f"{self.simulator_url}/api/environment/devices/blinds/{blinds_id}/control"
            
            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("success", False)
                else:
                    logger.error(f"Error controlling {blinds_id}: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending: {e}")
            return False
    
    def _create_ai_prompt(self, room: dict, state: dict) -> str:
        """Creates prompt for AI with room context and environment."""
        room_name = room.get("name", "Unknown")
        people_count = room.get("peopleCount", 0)
        blinds = room.get("blinds", {})
        current_blinds_state = blinds.get("state", "CLOSED")
        
        lights = room.get("lights", [])
        lights_on = any(light.get("state") == "ON" for light in lights)
        
        temp_sensor = room.get("temperatureSensor", {})
        room_temp = temp_sensor.get("temperature") if temp_sensor else None
        
        daylight = state.get("daylightIntensity", 1.0)
        external_temp = state.get("externalTemperature", 20.0)
        power_outage = state.get("powerOutage", False)
        
        meetings = room.get("scheduledMeetings", [])
        
        prompt = f"""You are an intelligent window blinds management system for an office building.

ROOM CONTEXT: {room_name}
- Number of people in room: {people_count}
- Current blinds state: {current_blinds_state}
- Lights on: {lights_on}
- Room temperature: {room_temp}°C (if available)
- External temperature: {external_temp}°C
- Daylight intensity: {daylight:.2f} (0.0 = night, 1.0 = full sun)
- Power outage: {power_outage}
- Scheduled meetings: {len(meetings)} meetings

RULES:
1. Power outage: OPEN blinds ONLY if there are people in the room (for safety)
2. Lights on: CLOSE blinds (energy saving)
3. Day + people in room: OPEN blinds (natural light)
4. Day + no people + high external temperature: CLOSE blinds (heat protection)
5. Night: CLOSE blinds (security)
6. Twilight: maintain current state or close for security

Respond in format: DECISION - REASON
Where DECISION is OPEN or CLOSED, and REASON is a short explanation (max 20 characters).
Example: "OPEN - day, people present" or "CLOSED - night, security"."""

        return prompt
    
    async def _ask_ai(self, prompt: str) -> Optional[tuple[str, str]]:
        """Ask AI for decision - uses Ollama."""
        if not self.use_ai:
            return None
        
        system_prompt = "You are an expert in energy management and comfort in office buildings. Respond in format: DECISION - REASON (e.g. 'OPEN - day, people present' or 'CLOSED - night, security')."
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "options": {"temperature": 0.3},
                        "stream": False  # Disable streaming for simpler handling
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                
                # Ollama may return streaming JSON (each line is a separate JSON)
                # or single JSON. We handle both cases.
                text = response.text.strip()
                
                # Try to parse as single JSON
                try:
                    result = response.json()
                    ai_response = result["message"]["content"].strip()
                except Exception:
                    # If still not working, it might be streaming - take the last line
                    lines = text.split('\n')
                    last_line = lines[-1] if lines else text
                    try:
                        result = json.loads(last_line)
                        ai_response = result["message"]["content"].strip()
                    except Exception:
                        # If still not working, try to extract text directly
                        # Sometimes Ollama returns only text without JSON
                        ai_response = text.strip()
                        # Remove any JSON characters
                        if ai_response.startswith('{'):
                            # Try to extract content from JSON
                            import re
                            match = re.search(r'"content"\s*:\s*"([^"]+)"', ai_response)
                            if match:
                                ai_response = match.group(1).strip()
            
            # Parse response: "DECISION - REASON" or just "DECISION"
            # AI may return multiple lines - extract only the line with DECISION
            logger.debug(f"Response from Ollama (raw): {ai_response[:300]}")
            
            # Split into lines and find the first line with OPEN or CLOSED
            lines = ai_response.strip().split('\n')
            decision_line = None
            for line in lines:
                line_upper = line.upper().strip()
                # Skip headers like "DECISION - REASON" and find actual decision
                if ("OPEN" in line_upper or "CLOSED" in line_upper) and not line_upper.startswith("DECISION") and not line_upper.startswith("REASON"):
                    decision_line = line.strip()
                    break
            
            # If not found in lines, use the whole response
            if not decision_line:
                decision_line = ai_response.strip()
            
            ai_response_upper = decision_line.upper()
            if "OPEN" in ai_response_upper:
                decision = "OPEN"
                # Extract reason if it exists
                if " - " in decision_line:
                    reason = decision_line.split(" - ", 1)[1].strip()
                    # Remove additional info if present
                    if "\n" in reason:
                        reason = reason.split("\n")[0].strip()
                    # Remove extra text after reason
                    reason = reason.split("REASON")[0].strip() if "REASON" in reason else reason
                elif ":" in decision_line:
                    reason = decision_line.split(":", 1)[1].strip()
                    if "\n" in reason:
                        reason = reason.split("\n")[0].strip()
                else:
                    reason = "AI decision"
                logger.debug(f"Recognized decision: {decision}, reason: {reason}")
                return (decision, reason)
            elif "CLOSED" in ai_response_upper:
                decision = "CLOSED"
                # Extract reason if it exists
                if " - " in decision_line:
                    reason = decision_line.split(" - ", 1)[1].strip()
                    # Remove additional info if present
                    if "\n" in reason:
                        reason = reason.split("\n")[0].strip()
                    # Remove extra text after reason
                    reason = reason.split("REASON")[0].strip() if "REASON" in reason else reason
                elif ":" in decision_line:
                    reason = decision_line.split(":", 1)[1].strip()
                    if "\n" in reason:
                        reason = reason.split("\n")[0].strip()
                else:
                    reason = "AI decision"
                logger.debug(f"Recognized decision: {decision}, reason: {reason}")
                return (decision, reason)
            else:
                logger.warning(f"AI provided unexpected response: {ai_response[:100]}")
                return None
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            logger.error(f"Error querying Ollama: {error_msg}")
            if 'response' in locals():
                logger.error(f"Response status: {response.status_code}")
                logger.error(f"Response from Ollama (first 500 characters): {response.text[:500]}")
            else:
                logger.error(f"No response from Ollama - exception: {type(e).__name__}")
            return None
    
    async def should_blinds_be_open(self, room: dict, state: dict) -> Optional[tuple[bool, str]]:
        """Decides if blinds should be open (uses only AI)."""
        if not self.use_ai:
            logger.error("AI is not available. Cannot make decisions.")
            return None
        
        room_name = room.get("name", "?")
        prompt = self._create_ai_prompt(room, state)
        ai_result = await self._ask_ai(prompt)
        
        if ai_result:
            decision, reason = ai_result
            should_open = decision == "OPEN"
            logger.debug(f"[{room_name}] AI decision: {decision} - {reason}")
            return should_open, f"AI: {reason}"
        
        logger.warning(f"[{room_name}] AI did not provide a decision. Skipping state change.")
        return None
    
    async def run_cycle(self):
        """One cycle of the agent."""
        state = await self.get_state()
        if not state:
            return
        
        rooms = state.get("rooms", [])
        
        for room in rooms:
            room_name = room.get("name", "?")
            blinds = room.get("blinds")
            
            if not blinds:
                continue
            
            blinds_id = blinds.get("id", "")
            current_state = blinds.get("state", "CLOSED")
            current_open = current_state == "OPEN"
            
            decision = await self.should_blinds_be_open(room, state)
            
            if decision is None:
                # AI did not provide a decision - skipping
                continue
            
            should_be_open, reason = decision
            
            # Open blinds
            if should_be_open and not current_open:
                success = await self.set_blinds(blinds_id, "OPEN")
                if success:
                    logger.info(f"⬆ OPENED {blinds_id} in {room_name} ({reason})")
                else:
                    logger.error(f"❌ Failed to open {blinds_id} in {room_name}")
            
            # Close blinds
            elif not should_be_open and current_open:
                success = await self.set_blinds(blinds_id, "CLOSED")
                if success:
                    logger.info(f"⬇ CLOSED {blinds_id} in {room_name} ({reason})")
                else:
                    logger.error(f"❌ Failed to close {blinds_id} in {room_name}")
            
            # State already matches AI decision
            elif should_be_open == current_open:
                logger.info(f"✓ {blinds_id} in {room_name} is already in the correct state ({current_state}) - {reason}")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WindowBlindsAgentAI - blinds agent with AI (Ollama)")
    parser.add_argument("simulator_url", nargs="?", default="http://localhost:8080",
                       help="URL simulator (default: http://localhost:8080)")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434",
                       help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--model", type=str, default="llama2",
                       help="Model Ollama to use (default: llama2)")
    
    args = parser.parse_args()
    
    agent = BlindsAgentAI(
        simulator_url=args.simulator_url,
        ollama_base_url=args.ollama_base_url,
        use_ai=True,
        model=args.model
    )
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
