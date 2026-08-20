"""The worker-size catalogue: read it from Girder, check it against Nova.

One rung per shape a submission may ask for. The catalogue lives in Girder as
``sivacor.worker_sizes`` (P0.3 of ``worker_sizing_plan.md``) because it has to be
readable by two processes that share no other config channel: ``girder-sivacor``
validates a submission against it and renders the picker from it, and this controller
boots the flavour it names.

**Read through Girder's model layer, not with a raw Mongo query, and the difference is
load-bearing.** ``Setting().get()`` applies ``SettingDefault``, so a deployment that has
never *written* the setting still gets ``girder_sivacor``'s own default. A raw
``find_one`` returns ``None`` there, and the failure is silent in the worst direction:
Girder validates submissions against a four-rung catalogue while the fleet believes
there are no rungs at all, so it boots nothing and every submission waits while the
numbers read healthy. That is exactly the trap P2's preflight found on the mirror,
where ``sivacor.worker_sizes`` has no document.

**But the model layer only helps if the default has been registered, and that took a
crash-loop to learn (mirror, 2026-08-20).** ``SettingDefault.defaults`` is not something
Girder knows a priori -- it is populated when ``girder_sivacor.settings`` is *imported*.
The Girder server imports it via ``SIVACORPlugin.load()``; this process has no plugin
load, so nothing imports it unless :func:`load` does. Reading through the model layer
with the owning module unimported gets the *worst* of both: the appearance of a default
and the behaviour of a raw read. See the imports in :func:`load`, which are both
mandatory for that reason.

This is why :func:`signals.targeted_assignment` may read raw and this may not: a boolean
whose default is ``False`` means "absent" and "off" are the same document, so nothing is
lost. A non-empty catalogue has no such luck.

Girder is imported lazily, inside the function that needs it, so :mod:`plan` and
:mod:`signals` stay testable in a bare virtualenv -- the same split :mod:`dispatch`
keeps.

**Why this module has no setting-key constant of its own while :mod:`signals` does.**
``signals.TARGETED_ASSIGNMENT_KEY`` is a hand-copied string with a "must match" comment,
and that is correct *there*: that module reads Mongo directly and must import nothing
from Girder, so it has no way to reference the real constant. Here the girder import is
already mandatory -- see :func:`load` -- so the key comes from ``PluginSettings`` and no
second copy exists to drift. Do not "tidy" a literal back in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .plan import SizeSpec

logger = logging.getLogger(__name__)

#: Nova advertises RAM in MiB, and the ``m3`` figures are exact binary -- verified live
#: 2026-08-20: 30 GiB = 30720, 60 = 61440, 125 = 128000, 250 = 256000. So the check in
#: :func:`validate` can be equality rather than a tolerance, and a mismatch really is a
#: mismatch rather than a rounding artefact.
MIB_PER_GIB = 1024


@dataclass(frozen=True)
class Rung:
    """One catalogue entry, as both processes see it.

    ``flavor`` is the only provider-specific string in the whole feature, and it stops
    here: S1 keeps the ``m3.*`` name out of the wire format, the exported workflow, the
    job document and the signed TRO, so nothing downstream can start depending on it.
    """

    memory_gb: int
    flavor: str
    vcpus: int
    #: Whether selecting this rung requires group membership (S5 guard 2). Carried so
    #: the catalogue round-trips faithfully, and read by ``girder-sivacor``'s picker in
    #: P4 -- **not** by the fleet. A gated rung boots exactly like any other; the gate
    #: is on who may ask for it.
    gated: bool = False

    @property
    def spec(self) -> SizeSpec:
        """The two numbers :func:`plan.decide` needs."""
        return SizeSpec(memory_gb=self.memory_gb, vcpus=self.vcpus)


def load(strict: bool = True) -> tuple[Rung, ...]:
    """The catalogue, oldest-fashioned way: whatever Girder says it is.

    ``strict`` off logs and drops a malformed entry instead of raising, for the one
    caller that would rather run on a partial catalogue than not at all. The default is
    on, because a catalogue this process cannot fully parse is a catalogue whose quota
    arithmetic would be wrong, and the controller refusing to start is far louder than a
    fleet that boots the wrong shapes.
    """
    # **Both imports are required, and the second one is not for the constant.**
    # `Setting().get()` falls back to `SettingDefault.defaults`, but that dict is
    # populated as an *import side effect* of `girder_sivacor.settings` -- in the Girder
    # server `SIVACORPlugin.load()` causes that import, and in this process nothing
    # does. Import only `girder.models.setting` and the fallback silently is not there:
    # on a deployment that never wrote the setting, `get()` returns None and the fleet
    # believes there are no rungs at all.
    #
    # That is not hypothetical -- it crash-looped the mirror's controller on 2026-08-20,
    # and it is why this module reads the key from `PluginSettings` rather than keeping
    # its own copy of the string. Importing the module that owns the default is the
    # thing that makes the default exist; taking the constant from it too means the
    # import cannot be mistaken for unused and dropped by a tidy-up.
    from girder.models.setting import Setting
    from girder_sivacor.settings import PluginSettings

    key = PluginSettings.WORKER_SIZES
    raw = Setting().get(key)
    if not raw:
        # Reachable only if girder_sivacor's SettingDefault is gone too, which means the
        # image is not the one this controller is supposed to be running on.
        raise RuntimeError(
            f"{key} is empty even through Girder's model layer, which "
            "applies the plugin's own default. Is girder_sivacor installed in this "
            "image? See P0.5 -- the controller and Girder share one build."
        )

    out: list[Rung] = []
    for entry in raw:
        try:
            rung = Rung(
                memory_gb=int(entry["memory_gb"]),
                flavor=str(entry["flavor"]),
                vcpus=int(entry["vcpus"]),
                gated=bool(entry.get("gated", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise RuntimeError(
                    f"malformed entry in {key}: {entry!r} ({exc}). "
                    "Girder validates this setting on write, so a bad entry here means "
                    "it was written straight into Mongo, bypassing the validator."
                ) from exc
            logger.warning("dropping malformed %s entry %r", key, entry)
            continue
        out.append(rung)

    if not out:
        raise RuntimeError(f"{key} has no usable entries")
    # Smallest first, so "the default rung" and "the cheapest rung" are the same lookup
    # everywhere and no caller has to remember to sort.
    return tuple(sorted(out, key=lambda r: r.memory_gb))


def specs(rungs) -> tuple[SizeSpec, ...]:
    """Just the arithmetic's view, for :attr:`plan.Limits.sizes`."""
    return tuple(r.spec for r in rungs)


def flavor_for(rungs, memory_gb: int | None, fallback: str) -> str:
    """Which OpenStack flavour a rung means.

    ``memory_gb`` of ``None`` is demand read from queue depth, which carries no size --
    only reachable while targeted assignment is off. Those get ``fallback``, i.e.
    ``SIVACOR_OS_FLAVOR``, which is precisely the pre-P3 behaviour for precisely the
    pre-P3 code path.
    """
    if memory_gb is None:
        return fallback
    for rung in rungs:
        if rung.memory_gb == memory_gb:
            return rung.flavor
    # decide() will not emit a rung that is not in the catalogue it was given, so this
    # is a programming error rather than a config one.
    raise RuntimeError(
        f"no catalogue entry for {memory_gb} GB; have "
        f"{[r.memory_gb for r in rungs]}"
    )


def validate(conn, rungs) -> None:
    """Resolve every rung's flavour against Nova and refuse to continue on a mismatch.

    **The duplication this checks is deliberate and the check is what makes it safe.**
    ``girder-sivacor`` has no OpenStack credential -- that is the property S3 exists to
    preserve -- so it cannot ask Nova for a flavour's shape, yet it must render
    ``· 16 cores`` in the picker and S6 needs vCPU for quota arithmetic. So ``vcpus`` is
    stored in the setting, and this process, the only one that *can* check it, does.
    Divergence becomes a boot failure here rather than silent drift there.

    Three failures it catches, in increasing order of how long they would otherwise
    take to find:

    * a flavour name that does not exist -- which at create time raises inside
      ``fleet.create_instance`` and, per P1, counts towards the circuit breaker, so
      three of them stop the **entire fleet**;
    * a ``vcpus`` figure that disagrees with Nova, which makes S6's quota arithmetic
      wrong in a direction nobody can see from the logs;
    * a ``memory_gb`` that disagrees, which means submissions are validated and billed
      against one number and run on another.
    """
    problems: list[str] = []
    for rung in rungs:
        flavor = conn.compute.find_flavor(rung.flavor)
        if flavor is None:
            problems.append(
                f"{rung.memory_gb} GB -> flavor {rung.flavor!r} does not exist in Nova"
            )
            continue
        if flavor.vcpus != rung.vcpus:
            problems.append(
                f"{rung.memory_gb} GB -> {rung.flavor} has {flavor.vcpus} vCPU, "
                f"catalogue says {rung.vcpus}"
            )
        want_mib = rung.memory_gb * MIB_PER_GIB
        if flavor.ram != want_mib:
            problems.append(
                f"{rung.memory_gb} GB -> {rung.flavor} has {flavor.ram} MiB "
                f"({flavor.ram / MIB_PER_GIB:.2f} GiB), catalogue implies {want_mib}"
            )
    if problems:
        raise RuntimeError(
            "worker size catalogue disagrees with Nova:\n  "
            + "\n  ".join(problems)
            + "\nFix the worker-size catalogue in Girder -- through the REST API, so "
            "the validator runs -- and restart this service, which reads it at startup."
        )
    logger.info(
        "worker sizes validated against Nova: %s",
        ", ".join(f"{r.memory_gb}GB={r.flavor}/{r.vcpus}vcpu" for r in rungs),
    )
