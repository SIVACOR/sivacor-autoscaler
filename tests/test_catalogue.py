"""Reading the worker-size catalogue, and checking it against Nova.

The validation here is not hygiene. `girder-sivacor` stores `vcpus` in the setting
because it has no OpenStack credential and cannot ask Nova (S3), so this process is the
only one that can check the duplication -- and an unknown flavour name reaching
`create_instance` raises, which counts towards the circuit breaker, so three of them
stop the entire fleet (P1).
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sivacor_autoscaler import catalogue
from sivacor_autoscaler.plan import SizeSpec

#: The catalogue as P3 ships it, with figures verified against live JS2 on 2026-08-20.
LADDER = [
    {"memory_gb": 30, "flavor": "m3.medium", "vcpus": 8, "gated": False},
    {"memory_gb": 60, "flavor": "m3.large", "vcpus": 16, "gated": False},
]


@pytest.fixture
def girder_setting(monkeypatch):
    """Stub Girder so `load()` can run in a bare virtualenv -- **including the coupling
    that crash-looped the mirror.**

    `catalogue.load` imports both modules lazily and by module path, which is what keeps
    `plan` and `signals` dependency-free, so the stubs go into `sys.modules`.

    What makes this stub worth reading: **the plugin default only exists once
    `girder_sivacor.settings` has been touched.** In reality that module registers it as
    an import side effect (`SettingDefault.defaults.update(...)` at module level); the
    Girder server triggers that via `SIVACORPlugin.load()` and this process triggers it
    from nowhere at all. So a `load()` that reads the setting through the model layer but
    never imports the module that owns the default gets the worst of both worlds -- the
    appearance of a fallback and the behaviour of a raw Mongo read -- and returns None on
    any deployment that never wrote the setting. That is what took the mirror's
    controller down on 2026-08-20.

    Here that is modelled with PEP 562 module `__getattr__`: the default lands in
    `defaults` only when something actually reads `PluginSettings` off the module. A stub
    that just returned a value, as the first version of this fixture did, passes happily
    either way -- which is exactly why the bug shipped.
    """

    def install(stored, *, plugin_default=None):
        defaults = {}

        class Setting:
            def get(self, key):
                if stored is not None:
                    return stored
                # Girder's real behaviour: no document -> SettingDefault.defaults.
                return defaults.get(key)

        setting_mod = ModuleType("girder.models.setting")
        setting_mod.Setting = Setting
        monkeypatch.setitem(sys.modules, "girder", ModuleType("girder"))
        monkeypatch.setitem(sys.modules, "girder.models", ModuleType("girder.models"))
        monkeypatch.setitem(sys.modules, "girder.models.setting", setting_mod)

        class PluginSettings:
            WORKER_SIZES = "sivacor.worker_sizes"

        plugin_mod = ModuleType("girder_sivacor.settings")

        def __getattr__(name):
            if name != "PluginSettings":
                raise AttributeError(name)
            if plugin_default is not None:
                defaults["sivacor.worker_sizes"] = plugin_default
            return PluginSettings

        plugin_mod.__getattr__ = __getattr__
        monkeypatch.setitem(sys.modules, "girder_sivacor", ModuleType("girder_sivacor"))
        monkeypatch.setitem(sys.modules, "girder_sivacor.settings", plugin_mod)
        return plugin_mod

    return install


def flavor(vcpus, ram_mib):
    return SimpleNamespace(vcpus=vcpus, ram=ram_mib)


def conn_with(**flavors):
    return SimpleNamespace(
        compute=SimpleNamespace(find_flavor=lambda name: flavors.get(name))
    )


NOVA = {
    "m3.medium": flavor(8, 30720),
    "m3.large": flavor(16, 61440),
}


# --- load ------------------------------------------------------------------


def test_the_catalogue_comes_back_smallest_first(girder_setting):
    """So "the default rung" and "the cheapest rung" are one lookup everywhere."""
    girder_setting(list(reversed(LADDER)))
    rungs = catalogue.load()
    assert [r.memory_gb for r in rungs] == [30, 60]
    assert [r.flavor for r in rungs] == ["m3.medium", "m3.large"]


def test_gated_round_trips_even_though_the_fleet_ignores_it():
    """S5 guard 2 is about who may *ask* for a rung; a gated rung boots like any other."""
    rung = catalogue.Rung(memory_gb=250, flavor="m3.2xl", vcpus=64, gated=True)
    assert rung.gated
    assert rung.spec == SizeSpec(memory_gb=250, vcpus=64)


def test_an_absent_setting_is_a_hard_error_not_an_empty_fleet(girder_setting):
    """The trap P2's preflight found on the mirror.

    Reading through Girder's model layer means SettingDefault covers a deployment that
    never wrote the setting -- so getting nothing back means girder_sivacor itself is
    missing, i.e. the wrong image. Returning () instead would leave Girder validating
    submissions against a catalogue the fleet believes is empty: it boots nothing and
    every submission waits while every number reads healthy.
    """
    girder_setting(None)
    with pytest.raises(RuntimeError, match="girder_sivacor"):
        catalogue.load()


def test_the_plugin_default_covers_a_deployment_that_never_wrote_the_setting(girder_setting):
    """THE regression. The mirror's controller crash-looped on this, 2026-08-20.

    No setting document -- which is the mirror's actual state, and what P2's preflight
    found -- so the whole claim that "reading through the model layer closes the
    absent-document trap" rests on `SettingDefault` having the plugin's default in it.
    It only does if `girder_sivacor.settings` has been imported, and in this process the
    only thing that can cause that import is `load()` itself.

    Drop `from girder_sivacor.settings import PluginSettings` from `load()` and this
    test fails with the exact error the mirror produced.
    """
    girder_setting(None, plugin_default=LADDER)

    rungs = catalogue.load()

    assert [r.memory_gb for r in rungs] == [30, 60]
    assert [r.flavor for r in rungs] == ["m3.medium", "m3.large"]


def test_a_written_setting_wins_over_the_plugin_default(girder_setting):
    """The ordinary case, and the one the mirror will be in once P3 is configured."""
    girder_setting([LADDER[1]], plugin_default=LADDER)
    assert [r.memory_gb for r in catalogue.load()] == [60]


def test_a_malformed_entry_refuses_to_start(girder_setting):
    """Girder validates on write, so a bad entry means someone wrote Mongo directly."""
    girder_setting([{"memory_gb": 30, "flavor": "m3.medium"}])  # no vcpus
    with pytest.raises(RuntimeError, match="malformed"):
        catalogue.load()


def test_strict_off_drops_the_bad_entry_and_keeps_the_rest(girder_setting):
    girder_setting([*LADDER, {"memory_gb": "moon"}])
    rungs = catalogue.load(strict=False)
    assert [r.memory_gb for r in rungs] == [30, 60]


# --- flavor_for ------------------------------------------------------------


def test_an_unsized_create_gets_the_configured_fallback():
    """Demand read from queue depth carries no size: pre-P3 path, pre-P3 flavour."""
    rungs = (catalogue.Rung(30, "m3.medium", 8),)
    assert catalogue.flavor_for(rungs, None, "m3.medium") == "m3.medium"


def test_a_sized_create_gets_its_rungs_flavour():
    rungs = (catalogue.Rung(30, "m3.medium", 8), catalogue.Rung(60, "m3.large", 16))
    assert catalogue.flavor_for(rungs, 60, "m3.medium") == "m3.large"


def test_a_rung_outside_the_catalogue_is_a_programming_error():
    """decide() cannot emit one, so reaching here means the two disagree."""
    rungs = (catalogue.Rung(30, "m3.medium", 8),)
    with pytest.raises(RuntimeError, match="no catalogue entry"):
        catalogue.flavor_for(rungs, 125, "m3.medium")


# --- validate --------------------------------------------------------------


def test_a_catalogue_that_matches_nova_validates():
    rungs = (catalogue.Rung(30, "m3.medium", 8), catalogue.Rung(60, "m3.large", 16))
    catalogue.validate(conn_with(**NOVA), rungs)  # does not raise


def test_a_flavour_that_does_not_exist_is_caught_before_anything_boots():
    """Otherwise it raises inside create_instance and three of those stop the fleet."""
    rungs = (catalogue.Rung(30, "m3.medum", 8),)  # typo
    with pytest.raises(RuntimeError, match="does not exist in Nova"):
        catalogue.validate(conn_with(**NOVA), rungs)


def test_a_vcpu_figure_that_disagrees_with_nova_is_caught():
    """The duplicated field. Wrong here makes S6's quota arithmetic silently wrong."""
    rungs = (catalogue.Rung(30, "m3.medium", 16),)
    with pytest.raises(RuntimeError, match="has 8 vCPU, catalogue says 16"):
        catalogue.validate(conn_with(**NOVA), rungs)


def test_a_memory_figure_that_disagrees_with_nova_is_caught():
    """`memory_gb` is what a submission is validated and billed against."""
    rungs = (catalogue.Rung(32, "m3.medium", 8),)
    with pytest.raises(RuntimeError, match="30720 MiB"):
        catalogue.validate(conn_with(**NOVA), rungs)


def test_every_problem_is_reported_at_once():
    """One restart per fix is a bad loop when the catalogue has four rungs."""
    rungs = (catalogue.Rung(30, "m3.medium", 99), catalogue.Rung(60, "nope", 16))
    with pytest.raises(RuntimeError) as exc:
        catalogue.validate(conn_with(**NOVA), rungs)
    assert "catalogue says 99" in str(exc.value)
    assert "does not exist in Nova" in str(exc.value)


def test_the_ram_check_is_exact_because_the_m3_figures_are_exact():
    """Verified live 2026-08-20: 30 GiB = 30720 MiB, 60 = 61440, 125 = 128000,
    250 = 256000. Every rung is memory_gb * 1024 with no remainder, so a tolerance
    would only ever hide a real mismatch."""
    for gb, mib in ((30, 30720), (60, 61440), (125, 128000), (250, 256000)):
        assert gb * catalogue.MIB_PER_GIB == mib
