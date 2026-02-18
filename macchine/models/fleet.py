"""Fleet registry: tracking which machine was at which site, when."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from macchine.models.core import Machine, Site


@dataclass
class FleetAssignment:
    """A machine assigned to a site for a date range."""

    machine: Machine
    site: Site
    start_date: datetime | None = None
    end_date: datetime | None = None
    trace_count: int = 0
    confidence: str = "high"  # high/medium/low based on source

    @property
    def duration_days(self) -> float | None:
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).total_seconds() / 86400
        return None


@dataclass
class FleetRegistry:
    """Registry of all machines, sites, and their assignments."""

    machines: dict[str, Machine] = field(default_factory=dict)  # keyed by serial_id
    sites: dict[str, Site] = field(default_factory=dict)  # keyed by site_id
    assignments: list[FleetAssignment] = field(default_factory=list)

    def get_machine_by_slug(self, slug: str) -> Machine | None:
        for m in self.machines.values():
            if m.slug == slug:
                return m
        return None

    def get_assignments_for_machine(self, serial_id: str) -> list[FleetAssignment]:
        return [a for a in self.assignments if a.machine.serial_id == serial_id]

    def get_assignments_for_site(self, site_id: str) -> list[FleetAssignment]:
        return [a for a in self.assignments if a.site.site_id == site_id]

    def add_machine(self, machine: Machine) -> None:
        self.machines[machine.serial_id] = machine

    def add_site(self, site: Site) -> None:
        self.sites[site.site_id] = site

    def add_assignment(self, assignment: FleetAssignment) -> None:
        self.assignments.append(assignment)

    def summary(self) -> str:
        lines = [
            f"Fleet: {len(self.machines)} machines, {len(self.sites)} sites, {len(self.assignments)} assignments",
            "",
            "Machines:",
        ]
        for m in sorted(self.machines.values(), key=lambda x: x.model_name):
            assignments = self.get_assignments_for_machine(m.serial_id)
            sites_str = ", ".join(a.site.name or a.site.site_id for a in assignments)
            lines.append(f"  {m.model_name} #{m.machine_number} ({m.serial_id}) -> {sites_str}")
        lines.append("")
        lines.append("Sites:")
        for s in sorted(self.sites.values(), key=lambda x: x.site_id):
            assignments = self.get_assignments_for_site(s.site_id)
            machines_str = ", ".join(a.machine.slug for a in assignments)
            total = s.json_count + s.dat_count
            lines.append(f"  {s.site_id} {s.name}: {total} files, machines: {machines_str}")
        return "\n".join(lines)
