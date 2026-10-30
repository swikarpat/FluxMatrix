"""Interactive client demo for FluxMatrix AI Inference Gateway.

Demonstrates:
- Dynamic semantic routing (Fast Tier vs Heavy Reasoning Tier)
- Real-time token streaming with SSE
- Live measurement of TTFT (Time to First Token) and TPOT (Time Per Output Token)
"""

import asyncio
import json
import time
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

GATEWAY_URL = "http://localhost:8080"


async def stream_query(client: httpx.AsyncClient, prompt: str, description: str):
    console.print(f"\n[bold cyan]=== Testing: {description} ===[/bold cyan]")
    console.print(f"[dim]User Prompt:[/dim] {prompt}\n")

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    received_text = []

    try:
        async with client.stream("POST", f"{GATEWAY_URL}/v1/chat/completions", json=payload) as response:
            if response.status_code != 200:
                console.print(f"[bold red]Error: Status {response.status_code}[/bold red]")
                return

            tier = response.headers.get("X-FluxMatrix-Tier", "unknown")
            model = response.headers.get("X-FluxMatrix-Model", "unknown")
            complexity = response.headers.get("X-FluxMatrix-Complexity", "N/A")
            trace_id = response.headers.get("X-Trace-ID", "N/A")

            console.print(
                f"[bold green]Routed Tier:[/bold green] [yellow]{tier}[/yellow] | "
                f"[bold green]Target Model:[/bold green] [magenta]{model}[/magenta] | "
                f"[bold green]Complexity Score:[/bold green] [cyan]{complexity}[/cyan]"
            )
            console.print(f"[dim]Trace ID:[/dim] {trace_id}\n")
            console.print("[bold yellow]Token Stream:[/bold yellow] ", end="")

            last_time = time.perf_counter()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now - start_time
                    else:
                        token_times.append(now - last_time)
                    last_time = now

                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            received_text.append(content)
                            console.print(f"[white]{content}[/white]", end="", highlight=False)

            total_duration = time.perf_counter() - start_time
            console.print("\n")

            # Latency Statistics Table
            avg_tpot = (sum(token_times) / len(token_times) * 1000.0) if token_times else 0.0
            ttft_ms = (first_token_time * 1000.0) if first_token_time else 0.0

            table = Table(title="Inference Performance Telemetry")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold green")
            table.add_column("SLA Threshold", style="dim")

            table.add_row("TTFT (Time to First Token)", f"{ttft_ms:.2f} ms", "< 80 ms")
            table.add_row("TPOT (Time Per Output Token)", f"{avg_tpot:.2f} ms/token", "< 25 ms/token")
            table.add_row("Total Output Tokens", str(len(received_text)), "N/A")
            table.add_row("Total Stream Duration", f"{total_duration:.2f} s", "N/A")

            console.print(table)

    except httpx.ConnectError:
        console.print("[bold red]Connection Error:[/bold red] FluxMatrix gateway is not running on http://localhost:8080.")
        console.print("[yellow]Start the gateway first via:[/yellow] python -m fluxmatrix.gateway.server\n")


async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Simple prompt (should route to Fast Tier)
        await stream_query(
            client,
            prompt="Summarize in one sentence: The team agreed to launch the beta release next Tuesday.",
            description="Simple Summarization Query (Expect Fast Quantized Tier)",
        )

        # Test 2: Complex reasoning prompt (should route to Heavy Reasoning Tier)
        await stream_query(
            client,
            prompt=(
                "Prove the correctness of an asynchronous non-blocking double-ended queue. "
                "Step by step, derive the memory orderings required for atomic compare-and-swap (CAS) "
                "to prevent ABA race conditions in C++20."
            ),
            description="Deep Algorithmic Reasoning Query (Expect Heavy Tier)",
        )


if __name__ == "__main__":
    asyncio.run(main())

