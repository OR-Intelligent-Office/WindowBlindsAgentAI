#!/usr/bin/env python3
"""
WindowBlindsAgentAI - agent rolet okiennych z integracją AI.

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
                 model: str = "llama2",
                 agent_id: str = "WindowBlindsAgent"):
        if not use_ai:
            raise ValueError("AI is required. WindowBlindsAgentAI cannot work without AI.")
        self.simulator_url = simulator_url
        self.session = None
        self.running = False
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.agent_id = agent_id
        self.last_message_timestamp: Optional[str] = None
        
        # Konfiguracja
        self.poll_interval = 2.0  # sekundy
        self.message_check_interval = 3.0  # sekundy - sprawdzaj wiadomości co 3 sekundy
        self.last_message_check = 0.0
        
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
        timeout = aiohttp.ClientTimeout(
            total=30.0,  # Total timeout: 30 seconds
            connect=10.0,  # Connection timeout: 10 seconds
            sock_read=30.0  # Socket read timeout: 30 seconds
        )
        self.session = aiohttp.ClientSession(timeout=timeout)
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
        """Gets the state of the environment from the simulator."""
        try:
            async with self.session.get(f"{self.simulator_url}/api/environment/state") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"Error getting state: {resp.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout connecting to simulator at {self.simulator_url}")
            return None
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"Cannot connect to simulator at {self.simulator_url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Connection error: {e}")
            return None
    
    async def set_blinds(self, blinds_id: str, state: str) -> bool:
        try:
            payload = {"state": state}
            url = f"{self.simulator_url}/api/environment/devices/blinds/{blinds_id}/control"
            
            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("success", False)
                else:
                    logger.error(f"Error setting blinds {blinds_id}: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return False
    
    async def send_message(self, to_agent: str, message: str, message_type: str = "INFORM") -> bool:
        """Sends a message in natural language to another agent."""
        try:
            payload = {
                "from": self.agent_id,
                "to": to_agent,
                "type": message_type,
                "content": message,
                "context": None
            }
            url = f"{self.simulator_url}/api/environment/agents/messages"
            
            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    success = result.get("success", False)
                    if success:
                        logger.info(f"📤 Message sent to {to_agent}: {message[:50]}...")
                    return success
                else:
                    logger.error(f"Error sending message to {to_agent}: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def get_messages(self) -> list[dict]:
        """Gets messages for this agent."""
        try:
            url = f"{self.simulator_url}/api/environment/agents/messages/{self.agent_id}"
            if self.last_message_timestamp:
                url = f"{self.simulator_url}/api/environment/agents/messages/{self.agent_id}/new?after={self.last_message_timestamp}"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    messages = await resp.json()
                    if messages and isinstance(messages, list):
                        # Update last message timestamp
                        if messages:
                            last_msg = max(messages, key=lambda m: m.get("timestamp", ""))
                            self.last_message_timestamp = last_msg.get("timestamp")
                        return messages
                    return []
                else:
                    logger.debug(f"Error getting messages: {resp.status}")
                    return []
        except Exception as e:
            logger.debug(f"Error getting messages: {e}")
            return []
    
    def _create_ai_prompt(self, room: dict, state: dict, received_messages: list[dict] = None) -> str:
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
        
        # Add messages context if any
        messages_context = ""
        if received_messages:
            messages_text = "\n".join([
                f"- From {msg.get('from', '?')}: {msg.get('content', '')[:100]}"
                for msg in received_messages[:3]  # Max 3 recent messages
            ])
            messages_context = f"\n\nRECEIVED MESSAGES FROM OTHER AGENTS:\n{messages_text}\n"
        
        prompt = f"""You are an intelligent system for managing window blinds in an office building.

CONTEXT OF THE ROOM: {room_name}
- Number of people in the room: {people_count}
- Current blinds state: {current_blinds_state}
- Lights on: {lights_on}
- Room temperature: {room_temp}°C (if available)
- External temperature: {external_temp}°C
- Daylight intensity: {daylight:.2f} (0.0 = night, 1.0 = full sun)
- Power outage: {power_outage}
- Scheduled meetings: {len(meetings)} meetings{messages_context}

DECISION HIERARCHY (apply in order, stop at first match):
1. Power outage: OPEN blinds ONLY if there are people in the room (for safety)
2. Lights on: CLOSE blinds (energy saving)
3. Day + people in the room: OPEN blinds (natural light)
4. Day + no people + high external temperature (>28°C): CLOSE blinds (heat protection)
5. Night: CLOSE blinds (safety)
6. Dusk: keep current state or close for safety

IMPORTANT: Return ONLY ONE decision following the hierarchy above. Do NOT provide multiple options or "OR" alternatives.

If you decide to CLOSE blinds due to high external temperature (>28°C), you should also inform the HeatingAgent about heat protection needs.
If you decide to OPEN blinds for natural light, you might inform HeatingAgent to reduce heating if temperature is high.

Answer in format: DECISION - REASON
Where DECISION is OPEN or CLOSED, and REASON is a short explanation (max 20 characters).
Example: "OPEN - day, people present" or "CLOSED - night, safety"."""

        return prompt
    
    async def _ask_ai(self, prompt: str) -> Optional[tuple[str, str]]:
        if not self.use_ai:
            return None
        
        system_prompt = "You are the expert in energy management and comfort in office buildings. Answer in the format: DECISION - REASON (e.g. 'OPEN - day, people in the room' or 'CLOSED - night, safety')."
        
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
                        "stream": False
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                
                # Ollama może zwracać streaming JSON (każda linia to osobny JSON)
                # lub pojedynczy JSON. Obsługujemy oba przypadki.
                text = response.text.strip()
                
                # Spróbuj parsować jako pojedynczy JSON
                try:
                    result = response.json()
                    ai_response = result["message"]["content"].strip()
                except Exception:
                    # Jeśli to nie działa, może być streaming - weź ostatnią linię
                    lines = text.split('\n')
                    last_line = lines[-1] if lines else text
                    try:
                        result = json.loads(last_line)
                        ai_response = result["message"]["content"].strip()
                    except Exception:
                        # Jeśli nadal nie działa, spróbuj wyciągnąć tekst bezpośrednio
                        # Czasami Ollama zwraca tylko tekst bez JSON
                        ai_response = text.strip()
                        # Usuń ewentualne znaki JSON
                        if ai_response.startswith('{'):
                            # Spróbuj wyciągnąć content z JSON
                            import re
                            match = re.search(r'"content"\s*:\s*"([^"]+)"', ai_response)
                            if match:
                                ai_response = match.group(1).strip()
            
            # Parsuj odpowiedź: "DECYZJA - POWÓD" lub tylko "DECYZJA"
            # Usuń wszystko po "OR" lub po drugiej opcji
            if "\n\nOR\n\n" in ai_response.upper() or "\nOR\n" in ai_response.upper():
                # Weź tylko pierwszą część przed "OR"
                ai_response = ai_response.split("\n\nOR\n\n")[0].split("\nOR\n")[0]
            
            # Podziel na linie i znajdź pierwszą linię z decyzją
            lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
            decision_line = None
            
            for line in lines:
                line_upper = line.upper()
                # Pomiń nagłówki i szukaj pierwszej linii z OPEN lub CLOSED
                if not line_upper.startswith("REASON") and not line_upper.startswith("DECISION"):
                    if "OPEN" in line_upper and "CLOSED" not in line_upper:
                        decision_line = line
                        break
                    elif "CLOSED" in line_upper:
                        decision_line = line
                        break
            
            # Jeśli nie znaleziono w liniach, użyj całej odpowiedzi
            if not decision_line:
                decision_line = ai_response.strip()
            
            # Wyciągnij decyzję i powód
            decision_line_upper = decision_line.upper()
            if "OPEN" in decision_line_upper and "CLOSED" not in decision_line_upper:
                decision = "OPEN"
                # Wyciągnij powód
                if " - " in decision_line:
                    reason = decision_line.split(" - ", 1)[1].strip()
                    # Usuń wszystko po "Reason:" lub po kropce jeśli jest długa
                    if "Reason:" in reason:
                        reason = reason.split("Reason:")[0].strip()
                    # Weź tylko pierwsze 20-30 znaków
                    if len(reason) > 30:
                        reason = reason[:30].strip()
                elif ":" in decision_line and not decision_line.upper().startswith("DECISION"):
                    reason = decision_line.split(":", 1)[1].strip()
                    if len(reason) > 30:
                        reason = reason[:30].strip()
                else:
                    reason = "day, people present"
                return (decision, reason)
            elif "CLOSED" in decision_line_upper:
                decision = "CLOSED"
                # Wyciągnij powód
                if " - " in decision_line:
                    reason = decision_line.split(" - ", 1)[1].strip()
                    # Usuń wszystko po "Reason:" lub po kropce jeśli jest długa
                    if "Reason:" in reason:
                        reason = reason.split("Reason:")[0].strip()
                    # Weź tylko pierwsze 20-30 znaków
                    if len(reason) > 30:
                        reason = reason[:30].strip()
                elif ":" in decision_line and not decision_line.upper().startswith("DECISION"):
                    reason = decision_line.split(":", 1)[1].strip()
                    if len(reason) > 30:
                        reason = reason[:30].strip()
                else:
                    reason = "night, safety"
                return (decision, reason)
            else:
                logger.warning(f"AI returned unexpected response: {ai_response[:100]}")
                return None
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            logger.debug(f"Response from Ollama: {response.text[:200] if 'response' in locals() else 'N/A'}")
            return None
    
    async def should_blinds_be_open(self, room: dict, state: dict, received_messages: list[dict] = None) -> Optional[tuple[bool, str]]:
        """Decides if blinds should be open (uses only AI)."""
        if not self.use_ai:
            logger.error("AI is not available. Cannot make decisions.")
            return None
        
        prompt = self._create_ai_prompt(room, state, received_messages)
        ai_result = await self._ask_ai(prompt)
        
        if ai_result:
            decision, reason = ai_result
            should_open = decision == "OPEN"
            
            # Automatically send messages to other agents in certain situations
            await self._send_notification_messages(room, state, decision, reason)
            
            return should_open, f"AI: {reason}"
        
        logger.warning("AI did not provide a decision. Skipping blind state change.")
        return None
    
    async def _send_notification_messages(self, room: dict, state: dict, decision: str, reason: str):
        """Sends notification messages to other agents when appropriate."""
        external_temp = state.get("externalTemperature", 20.0)
        room_temp = room.get("temperatureSensor", {}).get("temperature") if room.get("temperatureSensor") else None
        room_name = room.get("name", "Unknown")
        
        # If closing blinds due to high temperature, inform HeatingAgent
        if decision == "CLOSED" and external_temp > 28.0:
            message = f"Closed blinds in {room_name} due to high external temperature ({external_temp:.1f}°C) for heat protection."
            await self.send_message("HeatingAgent", message, "INFORM")
        
        # If opening blinds for natural light and temperature is high, suggest reducing heating
        elif decision == "OPEN" and "day" in reason.lower() and room_temp and room_temp > 23.0:
            message = f"Opened blinds in {room_name} for natural light. Room temperature is {room_temp:.1f}°C - consider reducing heating."
            await self.send_message("HeatingAgent", message, "INFORM")
    
    async def _check_and_process_messages(self):
        """Checks for new messages and processes them."""
        import time
        current_time = time.time()
        
        # Check messages periodically
        if current_time - self.last_message_check >= self.message_check_interval:
            self.last_message_check = current_time
            messages = await self.get_messages()
            
            if messages:
                for message in messages:
                    if message.get("to") == self.agent_id or message.get("to") == "broadcast":
                        logger.info(f"📨 Received message from {message.get('from', '?')}: {message.get('content', '')[:100]}")
                        await self._process_message(message)
    
    async def _process_message(self, message: dict):
        """Processes a received message using AI."""
        from_agent = message.get("from", "?")
        content = message.get("content", "")
        message_type = message.get("type", "INFORM")
        
        # Get current state
        state = await self.get_state()
        if not state:
            return
        
        # Create prompt for message processing
        prompt = f"""You are a window blinds management agent. You received a message from another agent.

MESSAGE:
From: {from_agent}
Type: {message_type}
Content: "{content}"

CURRENT ENVIRONMENT:
- External temperature: {state.get('externalTemperature', 20.0)}°C
- Daylight intensity: {state.get('daylightIntensity', 1.0):.2f}
- Power outage: {state.get('powerOutage', False)}
- Rooms with blinds: {len([r for r in state.get('rooms', []) if r.get('blinds')])}

Analyze the message and decide if you should:
1. Change blinds state based on the message
2. Send a response message back

If the message is about high temperature or heat protection, you might need to CLOSE blinds.
If the message is about low temperature or heating, you might need to OPEN blinds for natural light.

Answer in format: DECISION - REASON
Where DECISION is OPEN or CLOSED, and REASON references the message.
Example: "CLOSED - high temp from HeatingAgent" or "OPEN - heating request"."""

        # Ask AI for decision based on message
        ai_result = await self._ask_ai(prompt)
        
        if ai_result:
            decision, reason = ai_result
            should_open = decision == "OPEN"
            
            # Apply decision to all rooms with blinds
            rooms = state.get("rooms", [])
            for room in rooms:
                blinds = room.get("blinds")
                if not blinds:
                    continue
                
                blinds_id = blinds.get("id", "")
                current_state = blinds.get("state", "CLOSED")
                current_open = current_state == "OPEN"
                
                if should_open and not current_open:
                    success = await self.set_blinds(blinds_id, "OPEN")
                    if success:
                        logger.info(f"⬆ OPENED {blinds_id} in {room.get('name', '?')} (message from {from_agent}: {reason})")
                elif not should_open and current_open:
                    success = await self.set_blinds(blinds_id, "CLOSED")
                    if success:
                        logger.info(f"⬇ CLOSED {blinds_id} in {room.get('name', '?')} (message from {from_agent}: {reason})")
    
    async def run_cycle(self):
        """One cycle of agent operation."""
        # Check for messages first
        await self._check_and_process_messages()
        
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
            
            # Regular decision cycle (messages are already processed in _check_and_process_messages)
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
            
            # Close blinds
            elif not should_be_open and current_open:
                success = await self.set_blinds(blinds_id, "CLOSED")
                if success:
                    logger.info(f"⬇ CLOSED {blinds_id} in {room_name} ({reason})")

            else:
                logger.info(f"Blinds {blinds_id} in {room_name} are already in the desired state ({reason})")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WindowBlindsAgentAI - blinds agent with AI (Ollama)")
    parser.add_argument("simulator_url", nargs="?", default="http://localhost:8080",
                       help="Simulator URL (default: http://localhost:8080)")
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
