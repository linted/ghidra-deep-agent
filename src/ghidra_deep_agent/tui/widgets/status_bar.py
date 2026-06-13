from __future__ import annotations

from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static

from ghidra_deep_agent.tui.formatting import fmt_tokens


class StatusBar(Static):
    """Single-line status bar above the input: connections, timer, tokens, tools."""

    mcp_ok: reactive[bool] = reactive(True)
    db_ok: reactive[bool] = reactive(True)
    elapsed_seconds: reactive[int] = reactive(0)
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)
    current_context: reactive[int] = reactive(0)
    max_context: reactive[int] = reactive(200_000)
    active_tools: reactive[int] = reactive(0)
    flash_text: reactive[str] = reactive("")

    def on_mount(self) -> None:
        self._flash_timer: Timer | None = None
        self._refresh_status()

    def watch_mcp_ok(self, _old: bool, _new: bool) -> None:
        self._refresh_status()

    def watch_db_ok(self, _old: bool, _new: bool) -> None:
        self._refresh_status()

    def watch_elapsed_seconds(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_input_tokens(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_output_tokens(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_current_context(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_max_context(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_active_tools(self, _old: int, _new: int) -> None:
        self._refresh_status()

    def watch_flash_text(self, _old: str, _new: str) -> None:
        self._refresh_status()

    def flash(self, text: str, duration: float = 3.0) -> None:
        if self._flash_timer is not None:
            self._flash_timer.stop()
        self.flash_text = text
        self._flash_timer = self.set_timer(duration, self._clear_flash)

    def _clear_flash(self) -> None:
        self.flash_text = ""

    def _refresh_status(self) -> None:
        if self.flash_text:
            self.update(self.flash_text)
            return
        mcp = "[green]✓[/green]" if self.mcp_ok else "[red]✗[/red]"
        db = "[green]✓[/green]" if self.db_ok else "[red]✗[/red]"
        mins, secs = divmod(int(self.elapsed_seconds), 60)
        elapsed = f"{mins:02d}:{secs:02d}"
        toks_in = fmt_tokens(self.input_tokens)
        toks_out = fmt_tokens(self.output_tokens)
        ctx = self._format_context()
        sep = " [dim]│[/dim] "
        self.update(
            sep.join(
                [
                    f" mcp {mcp}  db {db}",
                    f"⏱ {elapsed}",
                    f"↓ {toks_in} in · ↑ {toks_out} out",
                    ctx,
                    f"⚙ {self.active_tools} active",
                ]
            )
        )

    def _format_context(self) -> str:
        max_ctx = max(self.max_context, 1)
        cur = self.current_context
        pct = (cur / max_ctx) * 100
        cur_s = fmt_tokens(cur)
        max_s = fmt_tokens(max_ctx)
        body = f"ctx {cur_s}/{max_s} ({pct:.0f}%)"
        if pct >= 85:
            return f"[red]{body}[/red]"
        if pct >= 75:
            return f"[yellow]{body}[/yellow]"
        return body
