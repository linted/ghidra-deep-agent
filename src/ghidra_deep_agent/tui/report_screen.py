"""Modal viewer for sub-agent final reports (``ctrl+o``).

Shows what each `task` run returned to the main agent — the sub-agent's
final message, which the main window never renders.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.rule import Rule
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, RichLog, Static

from ghidra_deep_agent.tui.formatting import fmt_duration
from ghidra_deep_agent.tui.messages import SubagentReport


class SubagentReportScreen(ModalScreen[None]):
    """Pick a sub-agent run (most recent first) and read its full report."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("ctrl+o", "close", "Close", show=False),
        Binding("ctrl+y", "copy_report", "Copy report"),
        Binding("pageup", "scroll_body(-1)", "Scroll up", show=False),
        Binding("pagedown", "scroll_body(1)", "Scroll down", show=False),
    ]

    def __init__(self, reports: list[SubagentReport]) -> None:
        super().__init__()
        self._reports = reports

    def compose(self) -> ComposeResult:
        with Vertical(id="report-box"):
            yield Static(
                "Sub-agent reports  —  ↑/↓ select · PgUp/PgDn scroll · "
                "Ctrl+Y copy · Esc close",
                id="report-title",
            )
            yield ListView(*self._rows(), id="report-list")
            yield RichLog(id="report-body", highlight=False, markup=False)

    def on_mount(self) -> None:
        self.query_one(ListView).focus()
        if self._reports:
            self._render_report(self._reports[0])

    def _rows(self) -> list[ListItem]:
        rows: list[ListItem] = []
        for i, report in enumerate(self._reports):
            marker = "[red]✗[/red]" if report.error else "[green]✓[/green]"
            label = (
                f"#{len(self._reports) - i}  {marker} "
                f"[dim]{fmt_duration(report.elapsed)}[/dim]  "
                f"{report.description[:60]}"
            )
            rows.append(ListItem(Label(label, markup=True)))
        return rows

    def _selected(self) -> SubagentReport | None:
        index = self.query_one(ListView).index
        if index is None or not (0 <= index < len(self._reports)):
            return None
        return self._reports[index]

    def _render_report(self, report: SubagentReport) -> None:
        body = self.query_one("#report-body", RichLog)
        body.clear()
        body.write(report.description)
        status = "[red]✗ errored[/red]" if report.error else "[green]✓ done[/green]"
        body.write(
            Rule(f"{status} in [dim]{fmt_duration(report.elapsed)}[/dim]", style="dim")
        )
        if report.text:
            body.write(Markdown(report.text))
        else:
            body.write("(empty report)")
        body.scroll_home(animate=False)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        report = self._selected()
        if report is not None:
            self._render_report(report)

    def action_scroll_body(self, direction: int) -> None:
        body = self.query_one("#report-body", RichLog)
        if direction < 0:
            body.scroll_page_up(animate=False)
        else:
            body.scroll_page_down(animate=False)

    def action_copy_report(self) -> None:
        report = self._selected()
        if report is None or not report.text:
            self.notify("Nothing to copy.", severity="warning")
            return
        self.app.copy_to_clipboard(report.text)
        self.notify("Report copied to clipboard.")

    def action_close(self) -> None:
        self.dismiss()
